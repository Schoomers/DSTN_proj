#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI for fleet-level (Table II) Stage II runs.

Example:
  python -m stage_II.fleet_cli --dataset_type SMART_ALIBABA --cohort_size 100 --num_cohorts 1 --out_name table2_smart_alibaba_fleet

If you prepared curated references for cohort-level text evaluation, pass:
  --refs_csv dataset/<...>/test_data/fleet_refs.csv

Required environment:
  export OPENAI_API_KEY=...
"""

from __future__ import annotations
import argparse
from pathlib import Path

from stage_II.config import Stage2Config, resolve_path
from stage_II.fleet_pipeline import FleetStage2Runner

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_type", required=True, help="SMART_ALIBABA | SMART_GOOGLE | SMART_WORKLOAD")
    p.add_argument("--input_csv", default=None, help="Override default CSV path for dataset_type.")
    p.add_argument("--tasks", default="predictive,descriptive,prescriptive,whatif", help="Comma list.")
    p.add_argument("--out_name", required=True, help="Output folder name under stage_II/runs.")
    p.add_argument("--cohort_size", type=int, default=100)
    p.add_argument("--num_cohorts", type=int, default=1)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--top_k_signals", type=int, default=2, help="How many per-drive SMART signals to include.")
    p.add_argument("--limit_rows", type=int, default=None, help="Optional: limit rows read from input CSV (debug).")
    p.add_argument("--refs_csv", default=None, help="Optional cohort-level references CSV for BLEU/ROUGE.")
    args = p.parse_args()

    cfg = Stage2Config(repo_root=Path("."))

    input_csv = Path(args.input_csv) if args.input_csv else cfg.dataset_type_to_csv.get(args.dataset_type)
    if input_csv is None:
        raise ValueError(f"Unknown dataset_type {args.dataset_type}. Provide --input_csv.")
    input_csv = resolve_path(cfg.repo_root, Path(input_csv))

    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]

    runner = FleetStage2Runner(cfg)
    out = runner.run(
        dataset_type=args.dataset_type,
        input_csv=input_csv,
        tasks=tasks,
        out_name=args.out_name,
        cohort_size=args.cohort_size,
        num_cohorts=args.num_cohorts,
        seed=args.seed,
        top_k_signals=args.top_k_signals,
        limit_rows=args.limit_rows,
        refs_csv=Path(args.refs_csv) if args.refs_csv else None,
    )
    print(f"[OK] Fleet run outputs written to: {out.run_dir}")
    print(f"Summary: {out.summary_json}")

if __name__ == "__main__":
    main()
