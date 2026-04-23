#!/usr/bin/env python3
"""
Select 19 SMART attributes + disk_id + ds from filtered Parquet and write daily CSVs.

Input (from previous step):
  dataset/alibaba_filtered/year=2018/model=MA1/*.parquet
  dataset/alibaba_filtered/year=2019/model=MA1/*.parquet
  ... for MB1, MB2, MA1, MA2, MC1, MC2

Output:
  dataset/alibaba/selected_attributes/MA1/2018/2018-01-01.csv
  dataset/alibaba/selected_attributes/MA1/2018/2018-01-02.csv
  ...
  dataset/alibaba/selected_attributes/MC2/2019/2019-12-31.csv

Run from repo root:
  python data_preparation/select_alibaba_attributes.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
except Exception:
    pa = None
    ds = None


SMART_FEATURES = [
    "r_5", "r_9", "r_12", "r_173", "r_174", "r_175", "r_177", "r_180", "r_181", "r_182",
    "r_183", "r_184", "r_187", "r_195", "r_197", "r_199", "r_233", "r_241", "r_242"
]

DEFAULT_MODELS = ["MA1", "MA2", "MB1", "MB2", "MC1", "MC2"]
DEFAULT_YEARS = ["2018", "2019"]


def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)


def normalize_ds_value(x) -> Optional[str]:
    """
    Convert ds/log_date values into a YYYY-MM-DD string suitable for filenames.
    Handles:
      - datetime-like
      - 'YYYY-MM-DD'
      - 'YYYYMMDD'
      - int like 20180123
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None

    # Pandas Timestamp / datetime
    try:
        if isinstance(x, (pd.Timestamp,)):
            return x.date().isoformat()
    except Exception:
        pass

    s = str(x).strip()

    # If it's already like YYYY-MM-DD (or has time)
    m = re.match(r"^(20\d{2})-(\d{2})-(\d{2})", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # If it's YYYYMMDD (string or int)
    m = re.match(r"^(20\d{2})(\d{2})(\d{2})$", s)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"

    # Try pandas parse as fallback
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.notna(ts):
            return ts.date().isoformat()
    except Exception:
        pass

    return None


def safe_daily_filename(ds_str: str) -> str:
    """Turn 'YYYY-MM-DD' into 'YYYY-MM-DD.csv' (also sanitizes just in case)."""
    ds_str = ds_str.strip()
    ds_str = re.sub(r"[^0-9\-]", "_", ds_str)
    return f"{ds_str}.csv"


def ensure_columns(df: pd.DataFrame, required: List[str]) -> pd.DataFrame:
    """Ensure all required columns exist; add missing as NA; reorder to required."""
    missing = [c for c in required if c not in df.columns]
    for c in missing:
        df[c] = pd.NA
    return df[required]


def write_daily_csvs(
    dataset_dir: Path,
    out_dir: Path,
    id_col: str,
    date_col: str,
    quiet: bool = False,
    batch_rows: int = 250_000,
) -> Dict[str, int]:
    """
    Read parquet dataset at dataset_dir, select columns, group by day, and write CSV per day.
    Returns dict: {ds: rows_written}
    """
    if ds is None:
        raise RuntimeError("pyarrow is required. Install with: pip install pyarrow")

    if not dataset_dir.exists():
        log(f"  - Skip (missing): {dataset_dir}", quiet=quiet)
        return {}

    out_dir.mkdir(parents=True, exist_ok=True)

    # Build Arrow dataset
    dset = ds.dataset(str(dataset_dir), format="parquet")
    cols = set(dset.schema.names)

    # Resolve date column: prefer date_col; fallback to log_date -> ds
    actual_date_col = None
    rename_to_ds = False
    if date_col in cols:
        actual_date_col = date_col
    elif "log_date" in cols:
        actual_date_col = "log_date"
        rename_to_ds = True
        log(f"  - Note: '{date_col}' not found; using 'log_date' as ds.", quiet=quiet)
    else:
        raise ValueError(
            f"Neither '{date_col}' nor 'log_date' found in {dataset_dir}. "
            f"Available columns (sample): {sorted(list(cols))[:40]}"
        )

    if id_col not in cols:
        raise ValueError(
            f"ID column '{id_col}' not found in {dataset_dir}. "
            f"Available columns (sample): {sorted(list(cols))[:40]}"
        )

    # Select only available smart features (we’ll also add missing as NA later to keep schema stable)
    available_features = [c for c in SMART_FEATURES if c in cols]
    needed_arrow_cols = [id_col, actual_date_col] + available_features

    # Streaming scan
    scanner = dset.scanner(columns=needed_arrow_cols, batch_size=batch_rows)

    required_final_cols = [id_col, "ds"] + SMART_FEATURES
    rows_written_by_day: Dict[str, int] = {}

    for rb in scanner.to_batches():
        df = rb.to_pandas()

        # Rename date column to ds if needed
        if rename_to_ds:
            df = df.rename(columns={actual_date_col: "ds"})
        else:
            if actual_date_col != "ds":
                df = df.rename(columns={actual_date_col: "ds"})

        # Normalize ds
        df["ds"] = df["ds"].apply(normalize_ds_value)

        # Drop rows without a valid date
        df = df[df["ds"].notna()]
        if df.empty:
            continue

        # Ensure all 21 columns exist and correct order
        df = ensure_columns(df, required_final_cols)

        # Group by day and append to that day’s file
        for day, g in df.groupby("ds", sort=False):
            day = str(day)
            fname = safe_daily_filename(day)
            fpath = out_dir / fname

            # Append; write header only once
            write_header = not fpath.exists()
            g.to_csv(fpath, mode="a", header=write_header, index=False)

            rows_written_by_day[day] = rows_written_by_day.get(day, 0) + len(g)

    return rows_written_by_day


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Select 19 SMART attributes + disk_id + ds from parquet and write daily CSVs."
    )
    p.add_argument("--repo-root", default=".", help="Repo root (default: current directory).")
    p.add_argument("--input-root", default="dataset/alibaba_filtered",
                   help="Input parquet root (default: dataset/alibaba_filtered).")
    p.add_argument("--output-root", default="dataset/alibaba/selected_attributes",
                   help="Output root (default: dataset/alibaba/selected_attributes).")
    p.add_argument("--models", nargs="*", default=DEFAULT_MODELS,
                   help="Models to process (default: MA1 MA2 MB1 MB2 MC1 MC2).")
    p.add_argument("--years", nargs="*", default=DEFAULT_YEARS,
                   help="Years to process (default: 2018 2019).")
    p.add_argument("--id-col", default="disk_id", help="Disk id column (default: disk_id).")
    p.add_argument("--date-col", default="ds", help="Date column (default: ds; fallback: log_date).")
    p.add_argument("--batch-rows", type=int, default=250_000, help="Arrow batch size (default: 250000).")
    p.add_argument("--quiet", action="store_true", help="Less logging.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite output model/year folders (DANGER: deletes existing daily CSVs).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if ds is None:
        raise RuntimeError("pyarrow is required. Install with: pip install pyarrow")

    repo_root = Path(args.repo_root).resolve()
    input_root = (repo_root / args.input_root).resolve()
    output_root = (repo_root / args.output_root).resolve()

    log(f"Repo root:   {repo_root}", quiet=args.quiet)
    log(f"Input root:  {input_root}", quiet=args.quiet)
    log(f"Output root: {output_root}", quiet=args.quiet)

    grand_total_files = 0
    grand_total_rows = 0

    for model in args.models:
        for year in args.years:
            in_dir = input_root / f"year={year}" / f"model={model}"
            out_dir = output_root / model / year

            if args.overwrite and out_dir.exists():
                # delete existing CSVs in that year folder
                for fp in out_dir.glob("*.csv"):
                    fp.unlink()

            log(f"\nProcessing model={model}, year={year}", quiet=args.quiet)
            log(f"  in:  {in_dir}", quiet=args.quiet)
            log(f"  out: {out_dir}", quiet=args.quiet)

            rows_by_day = write_daily_csvs(
                dataset_dir=in_dir,
                out_dir=out_dir,
                id_col=args.id_col,
                date_col=args.date_col,
                quiet=args.quiet,
                batch_rows=args.batch_rows,
            )

            num_days = len(rows_by_day)
            num_rows = sum(rows_by_day.values())
            grand_total_files += num_days
            grand_total_rows += num_rows

            log(f"  wrote {num_days} daily CSV files, {num_rows:,} rows", quiet=args.quiet)

    log(f"\nDONE. Total daily CSV files: {grand_total_files}, total rows: {grand_total_rows:,}",
        quiet=args.quiet)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"FATAL: {e}", file=sys.stderr)
        sys.exit(1)



"""
How to run
python data_preparation/select_alibaba_attributes.py

Only one model:
python data_preparation/select_alibaba_attributes.py --models MA1

"""