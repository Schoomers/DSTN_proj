#!/usr/bin/env python3
"""
stage2_pair_dataset_generator.py

Generate Stage II datasets for the (modality) pairs / triples described in your Table.

Supported dataset types
-----------------------
1) SMART
   - Sample from Alibaba or Google SMART-window datasets.
   - If Alibaba: drops the "app" column (per your requirement).

2) SMART_WORKLOAD
   - Sample from Alibaba SMART windows WITH "app" column (workload proxy).

3) SMART_ENV
   - Sample SMART (Alibaba or Google) and JOIN with env_effects.csv WITHOUT app column.

4) ENV_WORKLOAD
   - Sample env_effects.csv and JOIN with fio workloads (.fio) (e.g., fit_sample.fio or a folder).

5) SMART_ENV_WORKLOAD
   - Sample Alibaba SMART windows WITH "app" and JOIN with env_effects.csv WITH app.
   - (Workload = app column from Alibaba; does NOT add fio here unless you want it later.)

6) ENV
   - Sample directly from env_effects.csv

7) SMART_FT
   - Sample SMART and add flash_type column (SLC/MLC/TLC/QLC/PLC).

8) SMART_AL
   - Sample SMART and add controller policy columns (GC / WL / FTL mapping / ECC / refresh, etc.)

Inputs
------
- dataset type
- number of samples (n)
- optional: smart source (alibaba/google)
- optional: file paths override defaults

Outputs
-------
- A single CSV with n rows, written to your requested output path.

Assumed repo structure (defaults)
---------------------------------
dataset/
  alibaba/
    test_data/*.csv              # output from build_test_data_windows.py (has app)
  google/
    test_data/*.csv              # output from build_google_test_data_windows.py (may be a single file)
  env/
    env_effects.csv              # paper-only / effects table (NO windows)
    fio_workloads/*.fio          # generated fio jobs OR a single fit_sample.fio

Example runs (from repo root)
-----------------------------
# SMART only (Alibaba; app dropped)
python stage_II/stage2_pair_dataset_generator.py --dataset-type SMART --n 1000 --smart-source alibaba --out dataset/stage2/smart_only.csv

# SMART + Workload (Alibaba with app)
python stage_II/stage2_pair_dataset_generator.py --dataset-type SMART_WORKLOAD --n 1000 --out dataset/stage2/smart_workload.csv

# SMART + Env (Alibaba; env without app)
python stage_II/stage2_pair_dataset_generator.py --dataset-type SMART_ENV --n 500 --smart-source alibaba --env-csv dataset/env/env_effects.csv --out dataset/stage2/smart_env.csv

# Env + Workload (env + fio)
python stage_II/stage2_pair_dataset_generator.py --dataset-type ENV_WORKLOAD --n 500 --fio-path dataset/env/fio_workloads --out dataset/stage2/env_fio.csv

# SMART + Env + Workload (Alibaba app + env app)
python stage_II/stage2_pair_dataset_generator.py --dataset-type SMART_ENV_WORKLOAD --n 500 --env-csv dataset/env/env_effects.csv --out dataset/stage2/smart_env_workload.csv

Notes
-----
- Sampling is random but reproducible via --seed.
- If a source has fewer than n rows, the script will sample with replacement.
- Joins are done by random pairing (row-wise) NOT by keys.
- To avoid column name collisions when joining, env columns are prefixed with "env_" when needed;
  fio columns are prefixed with "fio_" when needed.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


DATASET_TYPES = [
    "SMART",
    "SMART_WORKLOAD",
    "SMART_ENV",
    "ENV_WORKLOAD",
    "SMART_ENV_WORKLOAD",
    "ENV",
    "SMART_FT",
    "SMART_AL",
]

FLASH_TYPES = ["SLC", "MLC", "TLC", "QLC", "PLC"]

GC_ALGOS = ["greedy", "cost_benefit", "windowed_greedy", "generational", "adaptive"]
WL_ALGOS = ["static", "dynamic", "hybrid"]
FTL_MAPPING = ["page_level", "block_level", "hybrid"]
REFRESH_POLICIES = ["fixed_interval", "adaptive", "retention_aware"]
ECC_SCHEMES = ["BCH", "LDPC", "LDPC_soft"]
READ_RETRY = ["enabled", "disabled"]
OP_PCT = [7, 12, 20, 28]


def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)


# -----------------------
# SMART loaders + sampling
# -----------------------

def find_csv_files(p: Path) -> List[Path]:
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.csv"))
    return []


def load_concat_csv(files: List[Path], usecols: Optional[List[str]] = None) -> pd.DataFrame:
    dfs = []
    for fp in files:
        df = pd.read_csv(fp, usecols=usecols) if usecols else pd.read_csv(fp)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def sample_df(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    if df.empty:
        return df
    replace = len(df) < n
    return df.sample(n=n, replace=replace, random_state=seed).reset_index(drop=True)


def load_smart_source(repo_root: Path, smart_source: str, smart_path: Optional[str], quiet: bool = False) -> pd.DataFrame:
    """
    smart_source:
      - alibaba: dataset/alibaba/test_data/*.csv  (has app)
      - google:  dataset/google/test_data/*.csv
    """
    if smart_path:
        p = (repo_root / smart_path).resolve()
    else:
        if smart_source == "alibaba":
            p = repo_root / "dataset" / "alibaba" / "test_data"
        elif smart_source == "google":
            p = repo_root / "dataset" / "google" / "test_data"
        else:
            raise ValueError("--smart-source must be 'alibaba' or 'google'")

    files = find_csv_files(p)
    if not files:
        raise FileNotFoundError(f"No SMART CSV files found at: {p}")

    log(f"SMART source={smart_source} files={len(files)} path={p}", quiet=quiet)
    df = load_concat_csv(files)
    return df


# -----------------------
# ENV loader + sampling
# -----------------------

def load_env(repo_root: Path, env_csv: Optional[str], quiet: bool = False) -> pd.DataFrame:
    p = (repo_root / env_csv).resolve() if env_csv else (repo_root / "dataset" / "env" / "env_effects.csv")
    if not p.exists():
        raise FileNotFoundError(f"Env CSV not found: {p}")
    df = pd.read_csv(p)
    log(f"ENV rows={len(df)} path={p}", quiet=quiet)
    return df


# -----------------------
# FIO loader + sampling
# -----------------------

@dataclass
class FioJob:
    name: str
    kv: Dict[str, str]
    text: str


def parse_fio_jobs_from_text(text: str) -> List[FioJob]:
    """
    Parse .fio content containing [global] and multiple [job] sections.
    Returns list of per-job objects.
    """
    lines = text.splitlines()
    jobs: List[FioJob] = []

    current_name: Optional[str] = None
    current_kv: Dict[str, str] = {}
    current_text_lines: List[str] = []

    def flush():
        nonlocal current_name, current_kv, current_text_lines
        if current_name and current_name.lower() != "global":
            jobs.append(FioJob(name=current_name, kv=dict(current_kv), text="\n".join(current_text_lines).strip() + "\n"))
        current_name = None
        current_kv = {}
        current_text_lines = []

    section_re = re.compile(r"^\s*\[(.+?)\]\s*$")
    kv_re = re.compile(r"^\s*([A-Za-z0-9_.-]+)\s*=\s*(.*?)\s*$")

    for ln in lines:
        m = section_re.match(ln)
        if m:
            flush()
            current_name = m.group(1).strip()
            current_text_lines = [ln]
            continue

        if current_name is None:
            # outside a section; ignore
            continue

        current_text_lines.append(ln)

        km = kv_re.match(ln)
        if km:
            k = km.group(1).strip()
            v = km.group(2).strip()
            current_kv[k] = v

    flush()
    return jobs


def load_fio_jobs(repo_root: Path, fio_path: Optional[str], quiet: bool = False) -> pd.DataFrame:
    """
    fio_path can be:
      - a directory containing *.fio
      - a single .fio file
    Default:
      dataset/env/fio_workloads (directory) if exists; else dataset/env/fio_workload; else dataset/env/fit_sample.fio.
    """
    if fio_path:
        p = (repo_root / fio_path).resolve()
    else:
        candidates = [
            repo_root / "dataset" / "env" / "fio_workloads",
            repo_root / "dataset" / "env" / "fio_workload",
            repo_root / "dataset" / "env" / "fit_sample.fio",
        ]
        p = next((c for c in candidates if c.exists()), None)
        if p is None:
            raise FileNotFoundError("Could not find fio workloads (set --fio-path).")

    files = find_csv_files(p)  # wrong for fio; handle separately
    fio_files: List[Path] = []
    if p.is_file():
        fio_files = [p]
    elif p.is_dir():
        fio_files = sorted(p.glob("*.fio"))
    else:
        raise FileNotFoundError(f"fio path not found: {p}")

    if not fio_files:
        raise FileNotFoundError(f"No .fio files found at: {p}")

    jobs: List[FioJob] = []
    for fp in fio_files:
        txt = fp.read_text(encoding="utf-8", errors="ignore")
        jobs.extend(parse_fio_jobs_from_text(txt))

    if not jobs:
        raise ValueError(f"No jobs parsed from fio files at: {p}")

    log(f"FIO jobs parsed={len(jobs)} from {len(fio_files)} file(s) at {p}", quiet=quiet)

    # Build a compact dataframe
    rows = []
    for j in jobs:
        row = {
            "job_name": j.name,
            "job_text": j.text,
        }
        # common keys (present if specified in job)
        for k in ["rw", "bs", "iodepth", "numjobs", "rwmixread", "random_distribution", "bssplit"]:
            if k in j.kv:
                row[k] = j.kv[k]
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------
# Joining helpers
# -----------------------

def safe_prefix_columns(df: pd.DataFrame, prefix: str, keep: Optional[List[str]] = None, collide_with: Optional[set] = None) -> pd.DataFrame:
    """
    Prefix df columns to avoid collisions with another dataframe.
    - keep: columns not to prefix
    - collide_with: set of column names to avoid; if a column exists in collide_with, prefix it
    """
    keep = set(keep or [])
    collide_with = set(collide_with or [])

    rename = {}
    for c in df.columns:
        if c in keep:
            continue
        if (c in collide_with) or (prefix and not c.startswith(prefix)):
            rename[c] = f"{prefix}{c}"
    return df.rename(columns=rename)


def pair_join(left: pd.DataFrame, right: pd.DataFrame, seed: int, right_prefix: str) -> pd.DataFrame:
    """
    Join by random pairing (row-wise): sample right to length(left), then concat columns.
    """
    if left.empty:
        return left
    if right.empty:
        return left

    r = sample_df(right, n=len(left), seed=seed)
    r = safe_prefix_columns(r, prefix=right_prefix, keep=[], collide_with=set(left.columns))
    out = pd.concat([left.reset_index(drop=True), r.reset_index(drop=True)], axis=1)
    return out


# -----------------------
# Enrichment helpers
# -----------------------

def add_flash_types(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["flash_type"] = rng.choice(FLASH_TYPES, size=len(df), replace=True)
    return df


def add_controller_algorithms(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df.copy()
    df["gc_algo"] = rng.choice(GC_ALGOS, size=len(df), replace=True)
    df["wear_leveling"] = rng.choice(WL_ALGOS, size=len(df), replace=True)
    df["ftl_mapping"] = rng.choice(FTL_MAPPING, size=len(df), replace=True)
    df["refresh_policy"] = rng.choice(REFRESH_POLICIES, size=len(df), replace=True)
    df["ecc_scheme"] = rng.choice(ECC_SCHEMES, size=len(df), replace=True)
    df["read_retry"] = rng.choice(READ_RETRY, size=len(df), replace=True)
    df["overprovision_pct"] = rng.choice(OP_PCT, size=len(df), replace=True)
    # Add a compact "policy_string" too (useful for LLM prompting)
    df["policy_string"] = (
        "gc=" + df["gc_algo"].astype(str)
        + ";wl=" + df["wear_leveling"].astype(str)
        + ";ftl=" + df["ftl_mapping"].astype(str)
        + ";refresh=" + df["refresh_policy"].astype(str)
        + ";ecc=" + df["ecc_scheme"].astype(str)
        + ";retry=" + df["read_retry"].astype(str)
        + ";op=" + df["overprovision_pct"].astype(str) + "%"
    )
    return df


# -----------------------
# Main generator
# -----------------------

def generate(
    repo_root: Path,
    dataset_type: str,
    n: int,
    seed: int,
    smart_source: str,
    smart_path: Optional[str],
    env_csv: Optional[str],
    fio_path: Optional[str],
    quiet: bool,
) -> pd.DataFrame:
    if dataset_type not in DATASET_TYPES:
        raise ValueError(f"Unsupported dataset type: {dataset_type}. Use one of {DATASET_TYPES}")

    # Load sources only as needed
    smart_df = None
    env_df = None
    fio_df = None

    if dataset_type in {"SMART", "SMART_WORKLOAD", "SMART_ENV", "SMART_ENV_WORKLOAD", "SMART_FT", "SMART_AL"}:
        smart_df = load_smart_source(repo_root, smart_source, smart_path, quiet=quiet)
        # Some runs only need a subset of columns; we keep full row as-is.
        smart_df = sample_df(smart_df, n=n, seed=seed)

    if dataset_type in {"ENV", "SMART_ENV", "ENV_WORKLOAD", "SMART_ENV_WORKLOAD"}:
        env_df = load_env(repo_root, env_csv, quiet=quiet)
        env_df = sample_df(env_df, n=n, seed=seed + 1)

    if dataset_type in {"ENV_WORKLOAD"}:
        fio_df = load_fio_jobs(repo_root, fio_path, quiet=quiet)
        fio_df = sample_df(fio_df, n=n, seed=seed + 2)

    # Apply dataset-specific transformations
    if dataset_type == "SMART":
        out = smart_df.copy()
        # per requirement: Alibaba SMART-only must NOT include app
        if smart_source == "alibaba" and "app" in out.columns:
            out = out.drop(columns=["app"])
        return out

    if dataset_type == "SMART_WORKLOAD":
        # per requirement: Alibaba with apps (workload proxy)
        out = smart_df.copy()
        # ensure app exists
        if "app" not in out.columns:
            out["app"] = "UNK"
        return out

    if dataset_type == "SMART_ENV":
        out = smart_df.copy()
        # In SMART+Env, you said env_effects.csv WITHOUT apps; also SMART only for Alibaba has no app
        if "app" in out.columns:
            out = out.drop(columns=["app"])
        env_local = env_df.copy()
        if "app" in env_local.columns:
            env_local = env_local.drop(columns=["app"])
        out = pair_join(out, env_local, seed=seed, right_prefix="env_")
        return out

    if dataset_type == "ENV_WORKLOAD":
        # Join env + fio
        env_local = env_df.copy()
        if "app" in env_local.columns:
            env_local = env_local.drop(columns=["app"])
        out = pair_join(env_local, fio_df, seed=seed, right_prefix="fio_")
        return out

    if dataset_type == "SMART_ENV_WORKLOAD":
        # SMART+Workload from Alibaba with app; join with env having app.
        out = smart_df.copy()
        if "app" not in out.columns:
            out["app"] = "UNK"

        env_local = env_df.copy()
        # Ensure env has app column, otherwise copy from SMART
        if "app" not in env_local.columns:
            env_local = env_local.copy()
            env_local["app"] = out["app"].sample(n=len(env_local), replace=True, random_state=seed).to_numpy()

        out = pair_join(out, env_local, seed=seed, right_prefix="env_")
        return out

    if dataset_type == "ENV":
        out = env_df.copy()
        return out

    if dataset_type == "SMART_FT":
        out = smart_df.copy()
        if smart_source == "alibaba" and "app" in out.columns:
            out = out.drop(columns=["app"])
        out = add_flash_types(out, seed=seed)
        return out

    if dataset_type == "SMART_AL":
        out = smart_df.copy()
        if smart_source == "alibaba" and "app" in out.columns:
            out = out.drop(columns=["app"])
        out = add_controller_algorithms(out, seed=seed)
        return out

    raise RuntimeError("Unhandled dataset type.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Stage II paired datasets (SMART/Env/Workload combinations).")
    p.add_argument("--dataset-type", required=True, choices=DATASET_TYPES, help="Dataset type as per Table 1.")
    p.add_argument("--n", type=int, required=True, help="Number of samples to output.")
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--seed", type=int, default=7, help="Random seed (default: 7).")
    p.add_argument("--repo-root", default=".", help="Repo root (default: current directory).")

    p.add_argument("--smart-source", default="alibaba", choices=["alibaba", "google"], help="SMART source (default: alibaba).")
    p.add_argument("--smart-path", default=None, help="Override SMART CSV path or directory (defaults to dataset/<source>/test_data).")

    p.add_argument("--env-csv", default=None, help="Env CSV path (default: dataset/env/env_effects.csv).")
    p.add_argument("--fio-path", default=None, help="Fio .fio file or directory (default: dataset/env/fio_workloads).")

    p.add_argument("--quiet", action="store_true", help="Less logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.n <= 0:
        raise ValueError("--n must be > 0")

    repo_root = Path(args.repo_root).resolve()
    out_path = (repo_root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate(
        repo_root=repo_root,
        dataset_type=args.dataset_type,
        n=args.n,
        seed=args.seed,
        smart_source=args.smart_source,
        smart_path=args.smart_path,
        env_csv=args.env_csv,
        fio_path=args.fio_path,
        quiet=args.quiet,
    )

    if df.empty:
        raise RuntimeError("Generated dataframe is empty (check your input paths).")

    df.insert(0, "dataset_type", args.dataset_type)

    df.to_csv(out_path, index=False)
    log(f"Wrote {len(df)} rows to: {out_path}", quiet=args.quiet)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)
