#!/usr/bin/env python3
"""Print the contents of responses.jsonl from a Stage 2 run directory.

Usage:
    python scripts/show_responses.py --run_dir stage_II/runs/demo_live
    python scripts/show_responses.py --run_dir stage_II/runs/demo_live --task diagnose
    python scripts/show_responses.py --run_dir stage_II/runs/demo_live --brief
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Print Stage 2 responses.jsonl")
    ap.add_argument("--run_dir", required=True,
                    help="Path to a Stage 2 run directory (contains responses.jsonl).")
    ap.add_argument("--task", default=None,
                    help="Optional: only show this task (diagnose/attribute/prescribe/whatif).")
    ap.add_argument("--sample", default=None,
                    help="Optional: only show this sample_id.")
    ap.add_argument("--brief", action="store_true",
                    help="Show only the top-level narrative fields, not full JSON.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    jsonl = run_dir / "responses.jsonl"
    if not jsonl.exists():
        print(f"ERROR: {jsonl} does not exist", file=sys.stderr)
        sys.exit(1)

    with jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            if args.task and r.get("task") != args.task:
                continue
            if args.sample and r.get("sample_id") != args.sample:
                continue

            print(f"\n=== {r.get('sample_id')}  /  {r.get('task')} ===")
            rj = r.get("response_json", {})

            if args.brief:
                for k in ("summary", "analysis", "scenario",
                          "primary_cause", "recommendations",
                          "counterfactual_statements", "atomic_claims"):
                    if k in rj:
                        print(f"\n-- {k} --")
                        print(json.dumps(rj[k], indent=2, ensure_ascii=False))
            else:
                print(json.dumps(rj, indent=2, ensure_ascii=False))

    # also print metrics summary if it's there
    summary = run_dir / "metrics_summary.json"
    if summary.exists():
        print(f"\n=== metrics_summary.json ===")
        print(summary.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
