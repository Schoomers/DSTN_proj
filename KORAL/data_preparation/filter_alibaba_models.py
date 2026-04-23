#!/usr/bin/env python3
"""
Filter Alibaba daily SSD SMART logs by model and write partitioned Parquet.

Repo layout (run from repo root):
  dataset/
    alibaba/
      smartlog2018ssd/*.csv
      smartlog2019ssd/*.csv
    google/
  data_preparation/
    filter_alibaba_models.py
  stage_II/

Output (default):
  dataset/alibaba_filtered/year=2018/model=MB1/part-000000.parquet
  dataset/alibaba_filtered/year=2019/model=MC2/part-000123.parquet
  dataset/alibaba_filtered/filter_summary.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:
    pa = None
    pq = None


TARGET_MODELS = {"MB1", "MB2", "MA1", "MA2", "MC1", "MC2"}


@dataclass
class Config:
    repo_root: Path
    input_root: Path
    year_folders: Tuple[str, str] = ("smartlog2018ssd", "smartlog2019ssd")
    output_root: Path = Path("dataset/alibaba_filtered")
    chunksize: int = 250_000
    model_column: Optional[str] = None  # auto-detect if None
    file_glob: str = "*.csv"
    sep: str = ","
    encoding: Optional[str] = None
    verbose: bool = True


def log(msg: str, verbose: bool = True) -> None:
    if verbose:
        print(msg, flush=True)


def list_daily_csvs(year_dir: Path, file_glob: str) -> List[Path]:
    files = sorted(year_dir.glob(file_glob))
    if not files:
        files = sorted(year_dir.rglob(file_glob))
    return [f for f in files if f.is_file()]


_date_patterns = [
    re.compile(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})"),  # 20180123 or 2018-01-23 or 2018_01_23
]


def infer_date_from_filename(path: Path) -> Optional[str]:
    name = path.stem
    for pat in _date_patterns:
        m = pat.search(name)
        if m:
            yyyy, mm, dd = m.group(1), m.group(2), m.group(3)
            return f"{yyyy}-{mm}-{dd}"
    return None


def detect_model_column(columns: Iterable[str]) -> str:
    cols = list(columns)
    lower = {c.lower(): c for c in cols}

    for candidate in ["model", "model_name", "ssd_model", "drive_model", "modelid", "model_id"]:
        if candidate in lower:
            return lower[candidate]

    modelish = [c for c in cols if "model" in c.lower()]
    if len(modelish) == 1:
        return modelish[0]
    if len(modelish) > 1:
        return sorted(modelish, key=lambda x: (len(x), x.lower()))[0]

    raise ValueError(
        "Could not auto-detect model column. "
        "Pass --model-column explicitly."
    )


class PartitionWriters:
    """Parquet writers keyed by (year, model)."""

    def __init__(self, output_root: Path):
        if pq is None or pa is None:
            raise RuntimeError("pyarrow is required. Install with: pip install pyarrow")
        self.output_root = output_root
        self._writers: Dict[Tuple[str, str], pq.ParquetWriter] = {}
        self._schemas: Dict[Tuple[str, str], pa.Schema] = {}
        self._part_index: Dict[Tuple[str, str], int] = {}

    def _next_path(self, year: str, model: str) -> Path:
        key = (year, model)
        idx = self._part_index.get(key, 0)
        self._part_index[key] = idx + 1
        out_dir = self.output_root / f"year={year}" / f"model={model}"
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"part-{idx:06d}.parquet"

    def write(self, df: pd.DataFrame, year: str, model: str) -> None:
        if df.empty:
            return
        key = (year, model)
        table = pa.Table.from_pandas(df, preserve_index=False)

        if key not in self._writers:
            path = self._next_path(year, model)
            self._schemas[key] = table.schema
            self._writers[key] = pq.ParquetWriter(
                where=str(path),
                schema=table.schema,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
        else:
            # If schema changes mid-stream, start a new file with the new schema.
            if table.schema != self._schemas[key]:
                self._writers[key].close()
                path = self._next_path(year, model)
                self._schemas[key] = table.schema
                self._writers[key] = pq.ParquetWriter(
                    where=str(path),
                    schema=table.schema,
                    compression="zstd",
                    use_dictionary=True,
                    write_statistics=True,
                )

        self._writers[key].write_table(table)

    def close_all(self) -> None:
        for w in self._writers.values():
            try:
                w.close()
            except Exception:
                pass
        self._writers.clear()
        self._schemas.clear()
        self._part_index.clear()


def filter_alibaba_logs(cfg: Config) -> None:
    if pq is None or pa is None:
        raise RuntimeError("Missing dependency: pyarrow. Install with: pip install pyarrow")

    cfg.output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "input_root": str(cfg.input_root),
        "output_root": str(cfg.output_root),
        "year_folders": list(cfg.year_folders),
        "target_models": sorted(TARGET_MODELS),
        "files_processed": 0,
        "rows_written_by_year_model": {},  # year -> model -> rows
        "errors": [],
    }

    writers = PartitionWriters(cfg.output_root)

    try:
        for year_folder in cfg.year_folders:
            year_dir = cfg.input_root / year_folder
            if not year_dir.exists():
                raise FileNotFoundError(f"Missing folder: {year_dir}")

            year_match = re.search(r"(20\d{2})", year_folder)
            year = year_match.group(1) if year_match else year_folder

            files = list_daily_csvs(year_dir, cfg.file_glob)
            log(f"[{year}] Found {len(files)} daily files in {year_dir}", verbose=cfg.verbose)
            if not files:
                continue

            model_col = cfg.model_column

            for i, fp in enumerate(files):
                date_str = infer_date_from_filename(fp)

                try:
                    # Step 1: detect model column once (fast header read)
                    if model_col is None:
                        head = pd.read_csv(fp, nrows=5, sep=cfg.sep, encoding=cfg.encoding, low_memory=False)
                        model_col = detect_model_column(head.columns)
                        log(f"Detected model column: '{model_col}'", verbose=cfg.verbose)

                    # Step 2: stream file by chunks
                    for chunk in pd.read_csv(
                        fp,
                        sep=cfg.sep,
                        encoding=cfg.encoding,
                        chunksize=cfg.chunksize,
                        low_memory=False,
                    ):
                        if model_col not in chunk.columns:
                            raise ValueError(
                                f"Model column '{model_col}' not found in {fp.name}. "
                                f"Columns: {list(chunk.columns)[:30]}"
                            )

                        # Step 3: filter to target models
                        sub = chunk[chunk[model_col].isin(TARGET_MODELS)].copy()
                        if sub.empty:
                            continue

                        # Step 4: attach date if we can infer it
                        if date_str is not None:
                            sub["log_date"] = date_str

                        # Step 5: write partitioned Parquet per model
                        for model, g in sub.groupby(model_col, sort=False):
                            writers.write(g, year=year, model=str(model))
                            ym = summary["rows_written_by_year_model"].setdefault(year, {})
                            ym[str(model)] = ym.get(str(model), 0) + int(len(g))

                    summary["files_processed"] += 1
                    if (i + 1) % 30 == 0:
                        log(f"[{year}] Processed {i+1}/{len(files)} files...", verbose=cfg.verbose)

                except Exception as e:
                    summary["errors"].append({"file": str(fp), "error": repr(e)})
                    log(f"ERROR on {fp}: {e}", verbose=True)

    finally:
        writers.close_all()
        (cfg.output_root / "filter_summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        log(f"Saved summary: {cfg.output_root / 'filter_summary.json'}", verbose=True)

    log("Done.", verbose=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter Alibaba daily SSD logs by model and write Parquet.")
    p.add_argument("--repo-root", default=".", help="Path to repo root (default: current directory).")
    p.add_argument("--input-root", default="dataset/alibaba", help="Alibaba dataset root (default: dataset/alibaba).")
    p.add_argument("--output-root", default="dataset/alibaba_filtered", help="Output root (default: dataset/alibaba_filtered).")
    p.add_argument("--chunksize", type=int, default=250_000, help="CSV read chunksize (default: 250000).")
    p.add_argument("--model-column", default=None, help="Model column name if auto-detect fails.")
    p.add_argument("--glob", default="*.csv", help="File glob for daily logs (default: *.csv).")
    p.add_argument("--quiet", action="store_true", help="Reduce logging.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    repo_root = Path(args.repo_root).resolve()
    input_root = (repo_root / args.input_root).resolve()
    output_root = (repo_root / args.output_root).resolve()

    cfg = Config(
        repo_root=repo_root,
        input_root=input_root,
        output_root=output_root,
        chunksize=args.chunksize,
        model_column=args.model_column,
        file_glob=args.glob,
        verbose=not args.quiet,
    )

    try:
        filter_alibaba_logs(cfg)
    except Exception as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        sys.exit(1)
"""
How to run
1. Install deps:
   python -m pip install pandas pyarrow

2. Run:
   python data_preparation/filter_alibaba_models.py

Optional (Optional (custom output path))
python data_preparation/filter_alibaba_models.py --output-root dataset/alibaba_filtered

"""