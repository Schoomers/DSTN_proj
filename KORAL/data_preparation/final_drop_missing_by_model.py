#!/usr/bin/env python3
"""
final_drop_missing_by_model.py

Last filter step: drop model-specific SMART attributes that are ~99.85% missing.

INPUT (from previous step):
  dataset/alibaba/selected_attributes/<MODEL>/<YEAR>/*.csv

OUTPUT (final daily CSVs):
  dataset/alibaba/<MODEL>/<YEAR>/*.csv

We always keep:
  - disk_id
  - ds

And we discard (per model):
  MA1: r_177, r_181, r_182, r_183, r_233, r_241, r_242
  MA2: r_173, r_177, r_180, r_181, r_182, r_195
  MB1: r_173, r_174, r_175, r_233
  MB2: r_173, r_174, r_175, r_233
  MC1: r_175, r_177, r_181, r_182, r_233, r_241, r_242
  MC2: r_175, r_177, r_181, r_182, r_233, r_241, r_242

This script processes each daily CSV independently (stream-friendly).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List


import pandas as pd


SMART_FEATURES: List[str] = [
    "r_5", "r_9", "r_12", "r_173", "r_174", "r_175", "r_177", "r_180", "r_181", "r_182",
    "r_183", "r_184", "r_187", "r_195", "r_197", "r_199", "r_233", "r_241", "r_242"
]

KEEP_ALWAYS: List[str] = ["disk_id", "ds"]

DROP_BY_MODEL: Dict[str, List[str]] = {
    "MA1": ["r_177", "r_181", "r_182", "r_183", "r_233", "r_241", "r_242"],
    "MA2": ["r_173", "r_177", "r_180", "r_181", "r_182", "r_195"],
    "MB1": ["r_173", "r_174", "r_175", "r_233"],
    "MB2": ["r_173", "r_174", "r_175", "r_233"],
    "MC1": ["r_175", "r_177", "r_181", "r_182", "r_233", "r_241", "r_242"],
    "MC2": ["r_175", "r_177", "r_181", "r_182", "r_233", "r_241", "r_242"],
}

DEFAULT_MODELS: List[str] = ["MA1", "MA2", "MB1", "MB2", "MC1", "MC2"]
DEFAULT_YEARS: List[str] = ["2018", "2019"]


def log(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg, flush=True)


def compute_keep_columns(model: str) -> List[str]:
    """Compute final column order for a model: disk_id, ds, then remaining SMART features."""
    drops = set(DROP_BY_MODEL.get(model, []))
    kept_smarts = [c for c in SMART_FEATURES if c not in drops]
    return KEEP_ALWAYS + kept_smarts


def process_one_file(
    in_path: Path,
    out_path: Path,
    keep_cols_ordered: List[str],
    quiet: bool = False,
) -> int:
    """
    Read one daily CSV, keep selected columns (safe if some columns are missing),
    and write to out_path. Returns number of rows written.
    """
    df = pd.read_csv(in_path, low_memory=False)

    # Validate required base columns exist
    for base in KEEP_ALWAYS:
        if base not in df.columns:
            raise ValueError(f"Missing required column '{base}' in {in_path}")

    # Keep only columns that exist; preserve desired order
    keep_existing = [c for c in keep_cols_ordered if c in df.columns]

    # If some SMART columns are missing, that's fine; we still write disk_id and ds at minimum.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out = df[keep_existing]
    df_out.to_csv(out_path, index=False)
    return int(len(df_out))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Drop model-specific missing SMART attributes from selected_attributes daily CSVs."
    )
    p.add_argument("--repo-root", default=".", help="Repo root (default: current directory).")
    p.add_argument(
        "--input-root",
        default="dataset/alibaba/selected_attributes",
        help="Input root containing <MODEL>/<YEAR> daily CSVs (default: dataset/alibaba/selected_attributes).",
    )
    p.add_argument(
        "--output-root",
        default="dataset/alibaba",
        help="Output root for final <MODEL>/<YEAR> daily CSVs (default: dataset/alibaba).",
    )
    p.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Models to process (default: MA1 MA2 MB1 MB2 MC1 MC2).",
    )
    p.add_argument(
        "--years",
        nargs="*",
        default=DEFAULT_YEARS,
        help="Years to process (default: 2018 2019).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output CSVs if present.",
    )
    p.add_argument("--quiet", action="store_true", help="Less logging.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    input_root = (repo_root / args.input_root).resolve()
    output_root = (repo_root / args.output_root).resolve()

    log(f"Repo root:   {repo_root}", quiet=args.quiet)
    log(f"Input root:  {input_root}", quiet=args.quiet)
    log(f"Output root: {output_root}", quiet=args.quiet)

    total_files = 0
    total_rows = 0
    total_skipped = 0

    for model in args.models:
        if model not in DROP_BY_MODEL:
            log(f"\nWARNING: Model '{model}' not in DROP_BY_MODEL; will keep all SMART features.", quiet=args.quiet)

        keep_cols = compute_keep_columns(model)

        for year in args.years:
            in_dir = input_root / model / year
            out_dir = output_root / model / year

            if not in_dir.exists():
                log(f"\nSkip missing input folder: {in_dir}", quiet=args.quiet)
                continue

            out_dir.mkdir(parents=True, exist_ok=True)
            files = sorted(in_dir.glob("*.csv"))

            log(f"\nProcessing model={model}, year={year}", quiet=args.quiet)
            log(f"  input:  {in_dir} ({len(files)} files)", quiet=args.quiet)
            log(f"  output: {out_dir}", quiet=args.quiet)

            for fp in files:
                out_fp = out_dir / fp.name
                if out_fp.exists() and not args.overwrite:
                    total_skipped += 1
                    continue

                try:
                    n = process_one_file(fp, out_fp, keep_cols, quiet=args.quiet)
                    total_files += 1
                    total_rows += n
                except Exception as e:
                    # Fail-fast for data integrity
                    raise RuntimeError(f"Failed on {fp}: {e}") from e

            log(f"  done: wrote {len(files) - total_skipped} files (skipped={total_skipped})", quiet=args.quiet)

    log(
        f"\nDONE. Files written: {total_files}, rows written: {total_rows:,}, skipped existing: {total_skipped}",
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
# 0) Ensure your repo layout is:
#    dataset/
#      alibaba/
#        selected_attributes/
#          MA1/2018/*.csv
#          MA1/2019/*.csv
#          ...
#    data_preparation/
#      final_drop_missing_by_model.py
#    stage_II/
#
# 1) Install dependencies:
#    python -m pip install pandas
#
# 2) Run the script:
#    python data_preparation/final_drop_missing_by_model.py
#
# 3) Output will be created as:
#    dataset/alibaba/MA1/2018/*.csv
#    dataset/alibaba/MA1/2019/*.csv
#    ...
#    dataset/alibaba/MC2/2019/*.csv
#
# 4) If you want to overwrite existing outputs:
#    python data_preparation/final_drop_missing_by_model.py --overwrite
#
# 5) Process only some models (example: MB1 and MB2):
#    python data_preparation/final_drop_missing_by_model.py --models MB1 MB2
#
# 6) Quiet mode:
#    python data_preparation/final_drop_missing_by_model.py --quiet
#
