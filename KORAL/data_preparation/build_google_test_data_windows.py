#!/usr/bin/env python3
"""
build_google_test_data_windows.py

Create N sampled 30-day windows from Google SSD dataset raw_data, with labels.

Inputs (repo layout)
--------------------
dataset/
  google/
    raw_data/
      errorlog.csv      # daily aggregates (may contain multiple samples per day per drive)
      swaplog.csv       # swap events
      badchip.csv       # repair/bad chip reports
    test_data/          # output folder (this script will create it if needed)

Output
------
A single CSV in:
  dataset/google/test_data/test_data.csv   (default; can override with --out)

Sampling
--------
- N windows total (user-defined via --n)
- Healthy:Failed ratio = 70:30 (default; can override with --healthy-ratio)
- One window per drive (no duplicates per drive)

Labels
------
- failure_time: earliest failure date per drive (swap date OR first status_dead day;
  optionally include badchip report date via --count-badchip)
- failure: 1 for failed-drive windows, 0 for healthy-drive windows

Window selection rule (aligned with your Alibaba logic)
-------------------------------------------------------
- For failed drives: select the *latest* 30-day window whose end date is within
  [0, window_to_failure_days] days before the failure_time (default 30 days).
- For healthy drives: select the *first* available 30-day window.

Features
--------
We map Google daily counters into an MB1-aligned r_* feature set (15 features),
matching the reference processing:
  r_241 ~ write_count
  r_242 ~ read_count
  r_187 ~ uncorrectable_error
  r_195 ~ correctable_error
  r_182 ~ erase_error
  r_181 ~ write_error
All other r_* in the MB1 feature set are filled with 0.0.

Flattening:
  <feature>_t00 ... <feature>_t29  (t00 oldest, t29 window end)

Run from repo root (examples are at bottom of file).
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ------------------------ Config ------------------------
MB1_FEATURES: List[str] = [
    "r_5", "r_9", "r_12", "r_177", "r_180",
    "r_181", "r_182", "r_183", "r_184", "r_187",
    "r_195", "r_197", "r_199", "r_241", "r_242",
]

# Google → MB1 mapping (best available approximation)
GOOGLE_TO_MB1_MAP: Dict[str, str] = {
    "write_count":         "r_241",
    "read_count":          "r_242",
    "uncorrectable_error": "r_187",
    "correctable_error":   "r_195",
    "erase_error":         "r_182",
    "write_error":         "r_181",
}

# Columns that represent per-day counts; when multiple entries exist per day, we SUM them
PER_DAY_COUNT_COLS: List[str] = [
    "read_count", "write_count", "erase_count",
    "correctable_error", "uncorrectable_error",
    "final_read_error", "final_write_error",
    "read_error", "write_error", "erase_error",
    "meta_error", "timeout_error", "response_error",
]

# Cumulatives/flags: take LAST or MAX when collapsing duplicates
CUMULATIVE_OR_STATIC_COLS: List[str] = [
    "cumulative_pe_cycle",
    "cumulative_bad_block_count",
    "factory_bad_block",
    "status_dead",
    "status_read_only",
]


def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)


def _find_errorlog(raw_dir: Path) -> Path:
    """
    Prefer errorlog.csv; fallback to google_data.csv (sample) if present.
    """
    p1 = raw_dir / "errorlog.csv"
    if p1.exists():
        return p1
    p2 = raw_dir / "google_data.csv"
    if p2.exists():
        return p2
    # fallback: try any csv that looks like errorlog
    for cand in raw_dir.glob("*.csv"):
        if "error" in cand.name.lower():
            return cand
    raise FileNotFoundError(f"Could not find errorlog.csv in {raw_dir}")


# ------------------------ Loaders ------------------------
def load_daily_data(raw_dir: Path) -> pd.DataFrame:
    path = _find_errorlog(raw_dir)
    d = pd.read_csv(path)

    if "timestamp_usec" not in d.columns:
        raise ValueError(f"{path.name} must include 'timestamp_usec'")

    # Normalize timestamp → day
    d["timestamp"] = pd.to_datetime(d["timestamp_usec"], unit="us", utc=True, errors="coerce")
    d = d.dropna(subset=["timestamp"])
    d["date"] = d["timestamp"].dt.floor("D")

    # Ensure model exists
    if "model" not in d.columns:
        d["model"] = np.nan

    # Coerce numeric
    for c in PER_DAY_COUNT_COLS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in CUMULATIVE_OR_STATIC_COLS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Ensure essential columns exist
    for c in PER_DAY_COUNT_COLS:
        if c not in d.columns:
            d[c] = 0.0
    for c in CUMULATIVE_OR_STATIC_COLS:
        if c not in d.columns:
            d[c] = 0.0

    # --- Deduplicate per (drive_id, date) BEFORE reindexing ---
    if "drive_id" not in d.columns:
        raise ValueError(f"{path.name} must include 'drive_id'")

    agg_dict = {c: "sum" for c in PER_DAY_COUNT_COLS}
    agg_dict.update({
        "cumulative_pe_cycle": "last",
        "cumulative_bad_block_count": "last",
        "factory_bad_block": "last",
    })
    agg_dict.update({"status_dead": "max", "status_read_only": "max"})

    def last_not_null(s: pd.Series):
        s2 = s.dropna()
        return s2.iloc[-1] if len(s2) else np.nan

    collapsed = (
        d.groupby(["drive_id", "date"], as_index=False)
         .agg({**agg_dict, "model": last_not_null})
    )

    # Fill NA to stable values
    for c in PER_DAY_COUNT_COLS + CUMULATIVE_OR_STATIC_COLS:
        collapsed[c] = collapsed[c].fillna(0)

    return collapsed


def load_swaplog(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "swaplog.csv"
    if not path.exists():
        return pd.DataFrame(columns=["drive_id", "swap_date"])
    s = pd.read_csv(path)
    if "time_of_swap" not in s.columns or "drive_id" not in s.columns:
        raise ValueError("swaplog.csv must include drive_id and time_of_swap")
    s["swap_date"] = pd.to_datetime(s["time_of_swap"], unit="us", utc=True, errors="coerce").dt.floor("D")
    s = s.dropna(subset=["swap_date"])
    return s[["drive_id", "swap_date"]]


def load_badchip(raw_dir: Path) -> pd.DataFrame:
    path = raw_dir / "badchip.csv"
    if not path.exists():
        return pd.DataFrame(columns=["drive_id", "report_date"])
    b = pd.read_csv(path)
    if "time_of_report" not in b.columns or "drive_id" not in b.columns:
        raise ValueError("badchip.csv must include drive_id and time_of_report")
    b["report_date"] = pd.to_datetime(b["time_of_report"], unit="us", utc=True, errors="coerce").dt.floor("D")
    b = b.dropna(subset=["report_date"])
    return b[["drive_id", "report_date"]]


# ------------------------ Processing ------------------------
def ensure_daily_continuity(df: pd.DataFrame) -> pd.DataFrame:
    """
    After deduping per-day, reindex each drive to a continuous daily range.
    Missing days:
      - per-day counts -> 0
      - cumulatives -> ffill then 0
      - flags -> 0
      - model -> ffill/bfill
    """
    out = []
    for did, g in df.groupby("drive_id"):
        g = g.sort_values("date")
        if g.empty:
            continue

        full_idx = pd.date_range(g["date"].min(), g["date"].max(), freq="D", tz="UTC")
        g = g.set_index("date").reindex(full_idx)
        g.index.name = "date"
        g["drive_id"] = did

        # model: forward/back fill
        g["model"] = g["model"].ffill().bfill()

        # cumulatives/static: ffill then 0
        for c in ["cumulative_pe_cycle", "cumulative_bad_block_count", "factory_bad_block"]:
            if c in g.columns:
                g[c] = g[c].ffill().fillna(0)

        # flags: missing -> 0
        for c in ["status_dead", "status_read_only"]:
            if c in g.columns:
                g[c] = g[c].fillna(0)

        # per-day counts: missing -> 0
        for c in PER_DAY_COUNT_COLS:
            if c in g.columns:
                g[c] = g[c].fillna(0)

        g = g.reset_index()
        out.append(g)

    return pd.concat(out, ignore_index=True) if out else df.copy()


def build_mb1_features(df: pd.DataFrame) -> pd.DataFrame:
    # init zeros for all MB1 features
    for r in MB1_FEATURES:
        df[r] = 0.0
    # map available google columns
    for g_col, r_name in GOOGLE_TO_MB1_MAP.items():
        if g_col in df.columns:
            df[r_name] = pd.to_numeric(df[g_col], errors="coerce").fillna(0.0)
    return df


def compute_failure_dates(
    daily: pd.DataFrame,
    swaplog: pd.DataFrame,
    badchip: pd.DataFrame,
    count_badchip: bool,
) -> pd.DataFrame:
    """
    failure_date = earliest of:
      - first day status_dead==1 in daily
      - earliest swap_date in swaplog
      - (optional) earliest report_date in badchip
    """
    dead = (
        daily[daily["status_dead"] == 1]
        .groupby("drive_id")["date"].min()
        .reset_index(name="dead_date")
    )
    sw = swaplog.groupby("drive_id")["swap_date"].min().reset_index()
    bc = badchip.groupby("drive_id")["report_date"].min().reset_index()

    keys = pd.DataFrame({"drive_id": daily["drive_id"].unique()})
    cand = keys.merge(dead, how="left").merge(sw, how="left").merge(bc, how="left")

    def earliest_failure(row):
        dates = []
        if pd.notna(row.get("dead_date")):
            dates.append(row["dead_date"])
        if pd.notna(row.get("swap_date")):
            dates.append(row["swap_date"])
        if count_badchip and pd.notna(row.get("report_date")):
            dates.append(row["report_date"])
        return min(dates) if dates else pd.NaT

    cand["failure_date"] = cand.apply(earliest_failure, axis=1)
    daily = daily.merge(cand[["drive_id", "failure_date"]], on="drive_id", how="left")
    return daily


# ------------------------ Window extraction ------------------------

@dataclass
class WindowRecord:
    disk_id: int
    ds_end: pd.Timestamp
    model: str
    app: str
    failure_time: Optional[pd.Timestamp]
    failure: int
    features: List[str]
    window_values: np.ndarray  # [30, F]


def select_one_window_per_drive(
    g: pd.DataFrame,
    feature_cols: List[str],
    window_days: int,
    window_to_failure_days: int,
    is_failed: bool,
    failure_date: Optional[pd.Timestamp],
) -> Optional[Tuple[pd.Timestamp, np.ndarray]]:
    """
    For one drive's daily series g (continuous):
      - If healthy: return earliest valid window (end index >= window_days-1)
      - If failed: return latest valid window whose end date is within [0..window_to_failure_days]
        days before failure_date (inclusive) and end_date <= failure_date.
    """
    g = g.sort_values("date")
    if len(g) < window_days:
        return None

    dates = g["date"].tolist()  # tz-aware UTC
    X = g[feature_cols].to_numpy(dtype=np.float32, copy=False)

    # valid end indices
    end_start = window_days - 1
    if not is_failed:
        i = end_start
        end_date = pd.Timestamp(dates[i]).normalize()
        window_vals = X[i - (window_days - 1): i + 1, :]
        return end_date, window_vals

    # Failed drive:
    if failure_date is None or pd.isna(failure_date):
        return None

    # Find the latest i such that:
    #   end_date <= failure_date AND 0 <= (failure_date - end_date).days <= window_to_failure_days
    best_i = None
    for i in range(len(dates) - 1, end_start - 1, -1):
        end_date = pd.Timestamp(dates[i]).normalize()
        if end_date > failure_date:
            continue
        ttf = (failure_date - end_date).days
        if 0 <= ttf <= window_to_failure_days:
            best_i = i
            break

    if best_i is None:
        return None

    end_date = pd.Timestamp(dates[best_i]).normalize()
    window_vals = X[best_i - (window_days - 1): best_i + 1, :]
    return end_date, window_vals


def flatten_window(rec: WindowRecord) -> Dict[str, object]:
    row: Dict[str, object] = {
        "disk_id": rec.disk_id,
        "ds": rec.ds_end.date().isoformat(),
        "model": rec.model,
        "app": rec.app,
        "failure_time": rec.failure_time.date().isoformat() if rec.failure_time is not None and pd.notna(rec.failure_time) else "",
        "failure": rec.failure,
    }

    vals = rec.window_values
    if vals.shape[0] != 30:
        raise ValueError(f"Window size is {vals.shape[0]}, expected 30")

    for fi, feat in enumerate(rec.features):
        for t in range(30):
            row[f"{feat}_t{t:02d}"] = float(vals[t, fi]) if not np.isnan(vals[t, fi]) else ""

    return row


# ------------------------ Main pipeline ------------------------

def build_windows_dataset(
    raw_dir: Path,
    out_csv: Path,
    n_total: int,
    healthy_ratio: float,
    seed: int,
    window_days: int = 30,
    window_to_failure_days: int = 30,
    count_badchip: bool = False,
    app_default: str = "UNK",
    quiet: bool = False,
) -> None:
    rng = random.Random(seed)

    # Step 1: load + preprocess daily data
    daily = load_daily_data(raw_dir)
    swaplog = load_swaplog(raw_dir)
    badchip = load_badchip(raw_dir)

    daily = ensure_daily_continuity(daily)
    daily = build_mb1_features(daily)
    daily = compute_failure_dates(daily, swaplog, badchip, count_badchip=count_badchip)

    # Step 2: candidate drive lists
    # Failed drives: have a failure_date
    failure_dates = (
        daily[["drive_id", "failure_date"]]
        .drop_duplicates("drive_id")
        .set_index("drive_id")["failure_date"]
    )

    drive_ids = daily["drive_id"].dropna().astype(int).unique().tolist()
    failed_ids = [int(did) for did in drive_ids if pd.notna(failure_dates.get(did, pd.NaT))]
    healthy_ids = [int(did) for did in drive_ids if pd.isna(failure_dates.get(did, pd.NaT))]

    rng.shuffle(failed_ids)
    rng.shuffle(healthy_ids)

    n_healthy_target = int(round(n_total * healthy_ratio))
    n_failed_target = n_total - n_healthy_target

    # Oversample candidates because not all drives will yield an eligible window
    oversample_factor = 5
    cand_failed = failed_ids[: min(len(failed_ids), max(n_failed_target * oversample_factor, n_failed_target))]
    cand_healthy = healthy_ids[: min(len(healthy_ids), max(n_healthy_target * oversample_factor, n_healthy_target))]

    candidate_set = set(cand_failed) | set(cand_healthy)
    log(f"Candidate drives: total={len(candidate_set)} (failed={len(cand_failed)}, healthy={len(cand_healthy)})", quiet=quiet)

    # Step 3: extract one window per candidate drive
    feature_cols = MB1_FEATURES

    healthy_windows: List[WindowRecord] = []
    failed_windows: List[WindowRecord] = []

    for did, g in daily[daily["drive_id"].isin(candidate_set)].groupby("drive_id"):
        did_int = int(did)
        fdate = failure_dates.get(did_int, pd.NaT)
        is_failed = pd.notna(fdate)

        sel = select_one_window_per_drive(
            g=g,
            feature_cols=feature_cols,
            window_days=window_days,
            window_to_failure_days=window_to_failure_days,
            is_failed=is_failed,
            failure_date=pd.Timestamp(fdate).normalize() if pd.notna(fdate) else None,
        )
        if sel is None:
            continue

        ds_end, window_vals = sel
        # model at window end: take last non-null
        m = g.sort_values("date")["model"].dropna()
        model_val = str(m.iloc[-1]) if len(m) else ""

        rec = WindowRecord(
            disk_id=did_int,
            ds_end=ds_end,
            model=model_val,
            app=app_default,
            failure_time=pd.Timestamp(fdate).normalize() if is_failed else None,
            failure=1 if is_failed else 0,
            features=feature_cols,
            window_values=window_vals,
        )

        if is_failed:
            failed_windows.append(rec)
        else:
            healthy_windows.append(rec)

    log(f"Extracted windows: healthy={len(healthy_windows)} failed={len(failed_windows)}", quiet=quiet)

    # Step 4: sample final N with ratio
    rng.shuffle(healthy_windows)
    rng.shuffle(failed_windows)

    if len(healthy_windows) < n_healthy_target:
        log(f"WARNING: requested healthy={n_healthy_target}, found={len(healthy_windows)}", quiet=quiet)
        n_healthy_target = len(healthy_windows)
        n_failed_target = min(n_failed_target, max(0, n_total - n_healthy_target))

    if len(failed_windows) < n_failed_target:
        log(f"WARNING: requested failed={n_failed_target}, found={len(failed_windows)}", quiet=quiet)
        n_failed_target = len(failed_windows)

    final_recs = healthy_windows[:n_healthy_target] + failed_windows[:n_failed_target]
    rng.shuffle(final_recs)

    # Step 5: write
    rows = [flatten_window(r) for r in final_recs]
    df_out = pd.DataFrame(rows)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)
    log(f"Saved {len(df_out)} windows to: {out_csv}", quiet=quiet)


# ------------------------ CLI ------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build N sampled 30-day windows for Google SSD dataset.")
    p.add_argument("--repo-root", default=".", help="Repo root (default: current directory).")
    p.add_argument("--raw-dir", default="dataset/google/raw_data", help="Raw data dir (default: dataset/google/raw_data).")
    p.add_argument("--n", type=int, required=True, help="Total number of windows to sample.")
    p.add_argument("--healthy-ratio", type=float, default=0.70, help="Healthy fraction (default: 0.70).")
    p.add_argument("--seed", type=int, default=7, help="Random seed (default: 7).")
    p.add_argument("--window-days", type=int, default=30, help="Window length in days (default: 30).")
    p.add_argument("--window-to-failure-days", type=int, default=30, help="Max days before failure (default: 30).")
    p.add_argument("--count-badchip", action="store_true", help="Include badchip report date as a failure signal.")
    p.add_argument("--app-default", default="UNK", help="Value to put in app column (default: UNK).")
    p.add_argument("--out", default=None, help="Output CSV (default: dataset/google/test_data/test_data.csv).")
    p.add_argument("--quiet", action="store_true", help="Less logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.n <= 0:
        raise ValueError("--n must be > 0")
    if not (0.0 < args.healthy_ratio < 1.0):
        raise ValueError("--healthy-ratio must be between 0 and 1")

    repo_root = Path(args.repo_root).resolve()
    raw_dir = (repo_root / args.raw_dir).resolve()
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")

    out_csv = (repo_root / args.out).resolve() if args.out else (repo_root / "dataset" / "google" / "test_data" / "test_data.csv")

    log(f"Repo root: {repo_root}", quiet=args.quiet)
    log(f"Raw dir:   {raw_dir}", quiet=args.quiet)
    log(f"Out CSV:   {out_csv}", quiet=args.quiet)

    build_windows_dataset(
        raw_dir=raw_dir,
        out_csv=out_csv,
        n_total=int(args.n),
        healthy_ratio=float(args.healthy_ratio),
        seed=int(args.seed),
        window_days=int(args.window_days),
        window_to_failure_days=int(args.window_to_failure_days),
        count_badchip=bool(args.count_badchip),
        app_default=str(args.app_default),
        quiet=bool(args.quiet),
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
# 2) Ensure the folder structure:
#    dataset/google/raw_data/errorlog.csv
#    dataset/google/raw_data/swaplog.csv
#    dataset/google/raw_data/badchip.csv
#
# 3) Run (example N=1000):
#    python stage_II/build_google_test_data_windows.py --n 1000
#
# 4) Output:
#    dataset/google/test_data/test_data.csv
#
# 5) Custom output filename:
#    python stage_II/build_google_test_data_windows.py --n 1000 --out dataset/google/test_data/google_test_data.csv
#
# 6) Include badchip report date as a failure signal:
#    python stage_II/build_google_test_data_windows.py --n 1000 --count-badchip
#
# 7) Reproducibility:
#    python stage_II/build_google_test_data_windows.py --n 1000 --seed 42
#
