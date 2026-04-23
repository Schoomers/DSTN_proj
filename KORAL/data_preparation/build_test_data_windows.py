#!/usr/bin/env python3
"""
build_test_data_windows.py

Build an N-sample dataset of 30-day windows for ONE model folder, using failure tags.

What this script does (step-by-step)
------------------------------------
1) Load failure tags from: dataset/alibaba/ssd_failure_tag.csv
   - Uses columns: model, failure_time, failure, app, disk_id
   - Keeps only failed disks (failure==1) for the given model code (A1/A2/B1/B2/C1/C2)
   - Uses the earliest failure_time per disk_id, and keeps its app.

2) Read the per-model daily SMART logs from your filtered data folder:
      dataset/alibaba/<MODEL_WITH_M>/<YEAR>/*.csv
   Example:
      dataset/alibaba/MB2/2018/*.csv
      dataset/alibaba/MB2/2019/*.csv

   Each daily CSV must contain at least:
      disk_id, ds, <SMART features...>

3) Build candidate disks:
   - Failed candidates: sample from failure tag disk_ids.
   - Healthy candidates: sample disk_ids from a few randomly chosen daily files (no failure tags).
     (We oversample candidates so we can still reach the requested N after filtering.)

4) Second pass over daily files (chronological):
   - For each candidate disk_id, track a rolling 30-day buffer (only for candidate disks).
   - Keep only buffers with 30 consecutive days.
   - For FAILED disks: keep the latest window whose end day is within 30 days of failure_time.
   - For HEALTHY disks: keep the first valid 30-day window found.

5) Randomly sample exactly N windows with healthy:failed ratio = 70:30 (configurable).
   - Output is ONE CSV in: dataset/alibaba/test_data/

Output format
-------------
One row = one 30-day window (flattened):
  disk_id, ds, model, app, failure_time, failure,
  <feature>_t00, <feature>_t01, ... <feature>_t29 for each feature

Where:
  - ds = window end date (YYYY-MM-DD)
  - t00 = oldest day in the 30-day window, t29 = newest (window end)

Notes
-----
- Failure tag models are A1/A2/B1/B2/C1/C2 (NO 'M').
- Your folder names are MA1/MA2/MB1/MB2/MC1/MC2 (WITH 'M').
- This script maps:
    MA1->A1, MA2->A2, MB1->B1, MB2->B2, MC1->C1, MC2->C2
- If you want a custom mapping, use --model-code explicitly.

Dependencies:
  pip install pandas numpy
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------
# Constants + mappings
# -----------------------

# Your folder names include 'M', failure tag model codes do not.
MODEL_FOLDER_TO_CODE = {
    "MA1": "A1",
    "MA2": "A2",
    "MB1": "B1",
    "MB2": "B2",
    "MC1": "C1",
    "MC2": "C2",
}

KEEP_BASE = ["disk_id", "ds"]


def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)


def parse_date(x) -> Optional[pd.Timestamp]:
    """Parse ds/failure_time into a normalized pandas.Timestamp (date only)."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    ts = pd.to_datetime(str(x), errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).normalize()


def infer_date_from_filename(p: Path) -> Optional[pd.Timestamp]:
    """Try to infer date from filename like 2018-01-02.csv or 20180102.csv."""
    s = p.stem
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", s)
    if not m:
        return None
    return pd.Timestamp(f"{m.group(1)}-{m.group(2)}-{m.group(3)}").normalize()


def list_daily_files(model_root: Path) -> List[Path]:
    """Collect and chronologically sort daily CSV files under model_root/<year>/*.csv."""
    files = []
    for year_dir in sorted([d for d in model_root.iterdir() if d.is_dir() and d.name.isdigit()]):
        files.extend(sorted(year_dir.glob("*.csv")))

    # Sort by inferred date; fallback to name
    def key(p: Path):
        d = infer_date_from_filename(p)
        return (d if d is not None else pd.Timestamp.min, p.name)

    files = sorted(files, key=key)
    return files


# -----------------------
# Failure tags loader
# -----------------------

@dataclass
class FailureInfo:
    failure_time: pd.Timestamp
    app: str


def load_failure_info(
    failure_tag_csv: Path,
    model_code: str,
) -> Dict[int, FailureInfo]:
    """
    Return disk_id -> FailureInfo for FAILED disks only (failure==1),
    using earliest failure_time per disk.
    """
    df = pd.read_csv(failure_tag_csv)

    required = {"model", "failure_time", "failure", "app", "disk_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ssd_failure_tag.csv missing required columns: {sorted(missing)}")

    df = df[df["model"] == model_code].copy()
    df = df[df["failure"] == 1].copy()

    if df.empty:
        return {}

    df["failure_time"] = pd.to_datetime(df["failure_time"].astype(str), errors="coerce")
    df = df.dropna(subset=["failure_time"])
    if df.empty:
        return {}

    # earliest failure per disk_id
    df = df.sort_values("failure_time")
    df = df.groupby("disk_id", as_index=False).first()[["disk_id", "failure_time", "app"]]

    out: Dict[int, FailureInfo] = {}
    for row in df.itertuples(index=False):
        out[int(row.disk_id)] = FailureInfo(
            failure_time=pd.Timestamp(row.failure_time).normalize(),
            app=str(row.app) if pd.notna(row.app) else "UNK",
        )
    return out


# -----------------------
# Candidate disk sampling
# -----------------------

def sample_failed_disks(
    failure_map: Dict[int, FailureInfo],
    n_failed_needed: int,
    oversample_factor: int,
    rng: random.Random,
) -> List[int]:
    failed_ids = list(failure_map.keys())
    rng.shuffle(failed_ids)
    k = min(len(failed_ids), max(n_failed_needed * oversample_factor, n_failed_needed))
    return failed_ids[:k]


def sample_healthy_disks_from_random_days(
    daily_files: List[Path],
    failed_ids: set,
    n_healthy_needed: int,
    oversample_factor: int,
    num_day_files_to_sample: int,
    rng: random.Random,
    quiet: bool = False,
) -> List[int]:
    """
    Sample healthy disk_ids from a few randomly chosen day files.
    This avoids scanning the whole dataset while still giving good variety.

    We oversample candidates so the second pass can still collect enough 30-day windows.
    """
    if not daily_files:
        return []

    k_days = min(num_day_files_to_sample, len(daily_files))
    chosen = rng.sample(daily_files, k=k_days)

    candidates = set()
    for fp in chosen:
        try:
            df = pd.read_csv(fp, usecols=["disk_id"])
        except Exception:
            df = pd.read_csv(fp)[["disk_id"]]

        ids = pd.Series(df["disk_id"]).dropna().astype(int).unique().tolist()
        for x in ids:
            if x not in failed_ids:
                candidates.add(int(x))

        log(f"  sampled disk_ids from {fp.name}: +{len(ids)} (candidates now {len(candidates)})", quiet=quiet)

    candidates = list(candidates)
    rng.shuffle(candidates)

    k = min(len(candidates), max(n_healthy_needed * oversample_factor, n_healthy_needed))
    return candidates[:k]


# -----------------------
# Window extraction pass
# -----------------------

@dataclass
class WindowRecord:
    disk_id: int
    ds_end: pd.Timestamp
    model_code: str
    app: str
    failure_time: Optional[pd.Timestamp]
    failure: int
    features: List[str]          # feature column names
    window_values: np.ndarray    # shape [30, F]


def extract_windows_for_candidates(
    daily_files: List[Path],
    candidate_disk_ids: set,
    failure_map: Dict[int, FailureInfo],
    model_code: str,
    quiet: bool = False,
    chunksize: int = 250_000,
) -> Tuple[List[WindowRecord], List[WindowRecord]]:
    """
    Second pass over daily files:
      - Track 30-day buffers only for candidate disks.
      - Pick one window per disk:
          FAILED disks: latest window with 0<=TTF<=30 days
          HEALTHY disks: first valid 30-day window
    Returns:
      (healthy_windows, failed_windows)
    """
    if not daily_files:
        return [], []

    # Determine feature columns from the first readable file header
    first_df = pd.read_csv(daily_files[0], nrows=1)
    if "disk_id" not in first_df.columns or "ds" not in first_df.columns:
        raise ValueError(f"{daily_files[0]} must include disk_id and ds columns.")
    feature_cols = [c for c in first_df.columns if c not in KEEP_BASE]

    log(f"Detected {len(feature_cols)} feature columns from first file.", quiet=quiet)

    # Per-disk rolling buffers: disk_id -> (last_date, list_of_rows)
    # Each row stored as (ds_timestamp, feature_values np.ndarray)
    buffers: Dict[int, List[Tuple[pd.Timestamp, np.ndarray]]] = {}
    last_seen_date: Dict[int, pd.Timestamp] = {}

    # Selected windows: disk_id -> WindowRecord
    selected_healthy: Dict[int, WindowRecord] = {}
    selected_failed: Dict[int, WindowRecord] = {}

    # For speed: local variables
    failed_ids = set(failure_map.keys())

    for fp in daily_files:
        # Early exit if we already have enough windows for all candidates (one per disk)
        if not candidate_disk_ids:
            break

        # Read file in chunks to handle large days
        reader = pd.read_csv(fp, chunksize=chunksize, low_memory=False)
        for chunk in reader:
            # Filter to candidate disks only
            if "disk_id" not in chunk.columns or "ds" not in chunk.columns:
                raise ValueError(f"{fp} missing disk_id or ds")

            # Some files might have extra columns, keep only what we need
            # and guard missing feature columns
            present_features = [c for c in feature_cols if c in chunk.columns]
            use_cols = ["disk_id", "ds"] + present_features
            sub = chunk[use_cols]

            sub = sub[sub["disk_id"].isin(candidate_disk_ids)]
            if sub.empty:
                continue

            # Parse ds
            sub["ds"] = pd.to_datetime(sub["ds"].astype(str), errors="coerce")
            sub = sub.dropna(subset=["ds"])
            if sub.empty:
                continue

            # Iterate rows (candidate disks are small; per-row is OK)
            for row in sub.itertuples(index=False):
                disk_id = int(row.disk_id)
                ds_ts = pd.Timestamp(row.ds).normalize()

                # Extract feature vector; if feature columns differ, align to full feature_cols
                row_dict = row._asdict()
                feat_vec = np.array([row_dict.get(c, np.nan) for c in feature_cols], dtype=np.float32)

                # Enforce consecutive days in buffer
                prev = last_seen_date.get(disk_id)
                if prev is not None and (ds_ts - prev).days != 1:
                    buffers[disk_id] = []
                last_seen_date[disk_id] = ds_ts

                buf = buffers.setdefault(disk_id, [])
                buf.append((ds_ts, feat_vec))
                if len(buf) > 30:
                    buf.pop(0)

                if len(buf) < 30:
                    continue

                # Candidate window
                ds_end = buf[-1][0]
                window_vals = np.stack([x[1] for x in buf], axis=0)  # [30, F]

                if disk_id in failed_ids:
                    finfo = failure_map[disk_id]
                    ttf = (finfo.failure_time - ds_end).days

                    # skip windows after failure
                    if ttf < 0:
                        # Once we pass failure, stop tracking this disk
                        if disk_id in candidate_disk_ids:
                            candidate_disk_ids.remove(disk_id)
                        buffers.pop(disk_id, None)
                        last_seen_date.pop(disk_id, None)
                        continue

                    # keep only windows within 30 days of failure
                    if 0 <= ttf <= 30:
                        selected_failed[disk_id] = WindowRecord(
                            disk_id=disk_id,
                            ds_end=ds_end,
                            model_code=model_code,
                            app=finfo.app,
                            failure_time=finfo.failure_time,
                            failure=1,
                            features=feature_cols,
                            window_values=window_vals,
                        )
                        # We do NOT remove disk yet; later windows may be even closer to failure.
                        # We remove once we hit failure day or pass it.
                        if ttf == 0:
                            # Can't get closer than day-of failure
                            if disk_id in candidate_disk_ids:
                                candidate_disk_ids.remove(disk_id)
                            buffers.pop(disk_id, None)
                            last_seen_date.pop(disk_id, None)
                    # else: too early, keep tracking
                else:
                    # healthy disk: keep the first valid window, then stop tracking
                    if disk_id not in selected_healthy:
                        selected_healthy[disk_id] = WindowRecord(
                            disk_id=disk_id,
                            ds_end=ds_end,
                            model_code=model_code,
                            app="UNK",
                            failure_time=None,
                            failure=0,
                            features=feature_cols,
                            window_values=window_vals,
                        )
                        if disk_id in candidate_disk_ids:
                            candidate_disk_ids.remove(disk_id)
                        buffers.pop(disk_id, None)
                        last_seen_date.pop(disk_id, None)

    healthy = list(selected_healthy.values())
    failed = list(selected_failed.values())
    return healthy, failed


# -----------------------
# Flatten + write output
# -----------------------

def flatten_window(rec: WindowRecord) -> Dict[str, object]:
    """
    Flatten a WindowRecord into one CSV row.

    Feature flattening:
      for each feature f and each time step t in [0..29]:
        column = f"{f}_t{t:02d}"   (t00 oldest, t29 newest)
    """
    row: Dict[str, object] = {
        "disk_id": rec.disk_id,
        "ds": rec.ds_end.date().isoformat(),
        "model": rec.model_code,
        "app": rec.app,
        "failure_time": rec.failure_time.date().isoformat() if rec.failure_time is not None else "",
        "failure": rec.failure,
    }

    vals = rec.window_values
    # Ensure correct shape
    if vals.shape[0] != 30:
        raise ValueError(f"Window size is {vals.shape[0]}, expected 30")

    for fi, feat in enumerate(rec.features):
        for t in range(30):
            row[f"{feat}_t{t:02d}"] = float(vals[t, fi]) if not np.isnan(vals[t, fi]) else ""

    return row


def save_windows_csv(
    out_csv: Path,
    healthy_recs: List[WindowRecord],
    failed_recs: List[WindowRecord],
    n_total: int,
    healthy_ratio: float,
    seed: int,
    quiet: bool = False,
) -> None:
    rng = random.Random(seed)

    n_healthy = int(round(n_total * healthy_ratio))
    n_failed = n_total - n_healthy

    if len(healthy_recs) < n_healthy:
        log(f"WARNING: requested {n_healthy} healthy windows, only found {len(healthy_recs)}", quiet=quiet)
        n_healthy = len(healthy_recs)
        n_failed = min(n_failed, max(0, n_total - n_healthy))

    if len(failed_recs) < n_failed:
        log(f"WARNING: requested {n_failed} failed windows, only found {len(failed_recs)}", quiet=quiet)
        n_failed = len(failed_recs)

    # Final sample
    rng.shuffle(healthy_recs)
    rng.shuffle(failed_recs)
    final = healthy_recs[:n_healthy] + failed_recs[:n_failed]
    rng.shuffle(final)

    rows = [flatten_window(r) for r in final]
    df_out = pd.DataFrame(rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    log(f"Saved {len(df_out)} windows to: {out_csv}", quiet=quiet)


# -----------------------
# CLI
# -----------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build N sampled 30-day windows CSV for one model folder.")
    p.add_argument("--repo-root", default=".", help="Repo root (default: current directory).")

    p.add_argument(
        "--model-folder",
        required=True,
        help="Path to per-model folder with year subfolders, e.g., dataset/alibaba/MB2",
    )
    p.add_argument(
        "--n",
        type=int,
        required=True,
        help="Total number of windows to sample (healthy+failed).",
    )
    p.add_argument(
        "--healthy-ratio",
        type=float,
        default=0.70,
        help="Healthy:failed ratio for sampling (default: 0.70).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed (default: 7).",
    )
    p.add_argument(
        "--failure-tag",
        default="dataset/alibaba/ssd_failure_tag.csv",
        help="Failure tag CSV path (default: dataset/alibaba/ssd_failure_tag.csv).",
    )
    p.add_argument(
        "--model-code",
        default=None,
        help="Failure-tag model code (A1/A2/B1/B2/C1/C2). If omitted, inferred from model folder name (MA1..MC2).",
    )
    p.add_argument(
        "--oversample-factor",
        type=int,
        default=5,
        help="Oversample candidate disks by this factor to improve chances (default: 5).",
    )
    p.add_argument(
        "--num-day-files-to-sample",
        type=int,
        default=8,
        help="How many daily files to use when sampling healthy disk IDs (default: 8).",
    )
    p.add_argument(
        "--chunksize",
        type=int,
        default=250_000,
        help="Read CSV chunksize for window extraction (default: 250000).",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output CSV path (default: dataset/alibaba/test_data/<MODEL>_n<N>.csv).",
    )
    p.add_argument("--quiet", action="store_true", help="Less logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    model_folder = (repo_root / args.model_folder).resolve()
    failure_tag_csv = (repo_root / args.failure_tag).resolve()

    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder not found: {model_folder}")
    if not failure_tag_csv.exists():
        raise FileNotFoundError(f"Failure tag CSV not found: {failure_tag_csv}")

    model_folder_name = model_folder.name  # e.g., MB2
    model_code = args.model_code
    if model_code is None:
        if model_folder_name not in MODEL_FOLDER_TO_CODE:
            raise ValueError(
                f"Cannot infer model-code from folder '{model_folder_name}'. "
                f"Pass --model-code explicitly (A1/A2/B1/B2/C1/C2)."
            )
        model_code = MODEL_FOLDER_TO_CODE[model_folder_name]

    n_total = int(args.n)
    if n_total <= 0:
        raise ValueError("--n must be > 0")
    if not (0.0 < args.healthy_ratio < 1.0):
        raise ValueError("--healthy-ratio must be between 0 and 1")

    out_csv = (repo_root / args.out).resolve() if args.out else (
        repo_root / "dataset" / "alibaba" / "test_data" / f"{model_folder_name}_n{n_total}.csv"
    )

    log(f"Repo root:        {repo_root}", quiet=args.quiet)
    log(f"Model folder:     {model_folder}", quiet=args.quiet)
    log(f"Model code:       {model_code} (from failure tags)", quiet=args.quiet)
    log(f"Failure tag CSV:  {failure_tag_csv}", quiet=args.quiet)
    log(f"Output CSV:       {out_csv}", quiet=args.quiet)

    # Step 1: failure map (failed disks only)
    failure_map = load_failure_info(failure_tag_csv, model_code)
    log(f"Failed disks in tags: {len(failure_map)}", quiet=args.quiet)

    # Step 2: list daily files
    daily_files = list_daily_files(model_folder)
    if not daily_files:
        raise FileNotFoundError(f"No daily CSV files found under: {model_folder}")
    log(f"Daily files found: {len(daily_files)}", quiet=args.quiet)
    log(f"Date range (by filename): {infer_date_from_filename(daily_files[0])} -> {infer_date_from_filename(daily_files[-1])}", quiet=args.quiet)

    # Step 3: decide counts
    n_healthy_target = int(round(n_total * args.healthy_ratio))
    n_failed_target = n_total - n_healthy_target
    log(f"Target windows: healthy={n_healthy_target}, failed={n_failed_target}", quiet=args.quiet)

    rng = random.Random(args.seed)

    # Step 4: sample candidate disk IDs (oversampled)
    candidate_failed = sample_failed_disks(failure_map, n_failed_target, args.oversample_factor, rng)
    failed_ids = set(candidate_failed)

    candidate_healthy = sample_healthy_disks_from_random_days(
        daily_files=daily_files,
        failed_ids=set(failure_map.keys()),  # exclude any known failed disk id
        n_healthy_needed=n_healthy_target,
        oversample_factor=args.oversample_factor,
        num_day_files_to_sample=args.num_day_files_to_sample,
        rng=rng,
        quiet=args.quiet,
    )

    # Avoid overlap just in case
    candidate_healthy = [x for x in candidate_healthy if x not in set(failure_map.keys())]

    candidate_disk_ids = set(candidate_failed) | set(candidate_healthy)
    log(f"Candidate disks total: {len(candidate_disk_ids)} (failed {len(candidate_failed)}, healthy {len(candidate_healthy)})", quiet=args.quiet)

    # Step 5: extract one window per candidate disk
    healthy_windows, failed_windows = extract_windows_for_candidates(
        daily_files=daily_files,
        candidate_disk_ids=candidate_disk_ids,
        failure_map=failure_map,
        model_code=model_code,
        quiet=args.quiet,
        chunksize=args.chunksize,
    )

    log(f"Extracted windows: healthy={len(healthy_windows)}, failed={len(failed_windows)}", quiet=args.quiet)

    # Step 6: sample final N with ratio and save
    save_windows_csv(
        out_csv=out_csv,
        healthy_recs=healthy_windows,
        failed_recs=failed_windows,
        n_total=n_total,
        healthy_ratio=args.healthy_ratio,
        seed=args.seed,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)


# ===========================
# HOW TO RUN (from repo root)
# ===========================
#
# 1) Install dependencies:
#    python -m pip install pandas numpy
#
# 2) Make sure your data exists:
#    - Failure tags:
#        dataset/alibaba/ssd_failure_tag.csv
#    - Filtered per-model daily logs:
#        dataset/alibaba/MB2/2018/*.csv
#        dataset/alibaba/MB2/2019/*.csv
#      (Similarly for MA1, MA2, MB1, MC1, MC2)
#
# 3) Run for one model (example MB2) and N=1000 windows:
#    python stage_II/build_test_data_windows.py --model-folder dataset/alibaba/MB2 --n 1000
#
# 4) Output will be a single CSV:
#    dataset/alibaba/test_data/MB2_n1000.csv
#
# 5) If you want a custom output name:
#    python stage_II/build_test_data_windows.py --model-folder dataset/alibaba/MB2 --n 1000 --out dataset/alibaba/test_data/test_data.csv
#
# 6) If the folder name cannot be mapped to A1/A2/B1/B2/C1/C2, pass model code explicitly:
#    python stage_II/build_test_data_windows.py --model-folder dataset/alibaba/MB2 --n 1000 --model-code B2
#
# 7) To change the healthy:failed ratio (default 70:30):
#    python stage_II/build_test_data_windows.py --model-folder dataset/alibaba/MB2 --n 1000 --healthy-ratio 0.7
#
# 8) For reproducibility:
#    python stage_II/build_test_data_windows.py --model-folder dataset/alibaba/MB2 --n 1000 --seed 42
#
