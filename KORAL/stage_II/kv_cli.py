#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI for KV-SSD Stage II diagnosis pipeline.

Input can be:
  --input file.jsonl   (preferred; one KV telemetry record per line)
  --input file.csv     (each row is one window)
  --input file.json    (single record or array)

Upstream adapters that produce this input:
  - stage_II.adapters.mqsim.parse_mqsim_xml
  - stage_II.adapters.rocksdb_stats.parse_rocksdb_log
  - stage_II.adapters.kvpack.parse_kvpack_jsonl

LLM backends: openai (default), ollama (local), none (plumbing-only dry run).
"""

from __future__ import annotations
import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import List

# Allow `python3 stage_II/kv_cli.py ...` from anywhere: make KORAL/ importable.
_KORAL_DIR = Path(__file__).resolve().parent.parent
if str(_KORAL_DIR) not in sys.path:
    sys.path.insert(0, str(_KORAL_DIR))

from stage_II.config import Stage2Config, resolve_path
from stage_II.kv_pipeline import KVStage2Runner, TASKS


def _run_adapter(kind: str, path: Path) -> List[dict]:
    if kind == "mqsim":
        from stage_II.adapters.mqsim import parse_mqsim_xml, parse_mqsim_dir
        if path.is_dir():
            return [parse_mqsim_dir(path)]
        return [parse_mqsim_xml(path)]
    if kind == "rocksdb":
        from stage_II.adapters.rocksdb_stats import parse_rocksdb_log
        return [parse_rocksdb_log(path)]
    if kind == "kvpack":
        from stage_II.adapters.kvpack import parse_kvpack_json, parse_kvpack_jsonl
        if path.suffix.lower() == ".jsonl":
            return parse_kvpack_jsonl(path)
        return parse_kvpack_json(path)
    raise ValueError(f"Unknown --adapter kind: {kind}")


def main():
    ap = argparse.ArgumentParser(description="KORAL Stage II — KV-SSD diagnosis pipeline")
    ap.add_argument("--input", type=str, required=False,
                    help="Path to a JSONL/CSV/JSON of KV telemetry rows.")
    ap.add_argument("--adapter", type=str, choices=["mqsim", "rocksdb", "kvpack"],
                    help="Use an adapter to convert a simulator/log file into KV telemetry rows first.")
    ap.add_argument("--adapter_input", type=str,
                    help="Source file/dir for the chosen --adapter.")
    ap.add_argument("--tasks", type=str, default=",".join(TASKS),
                    help=f"Comma-separated subset of {','.join(TASKS)}.")
    ap.add_argument("--limit_rows", type=int, default=None)
    ap.add_argument("--out_name", type=str, default=None)
    ap.add_argument("--llm_backend", type=str, default="openai",
                    choices=["openai", "ollama", "none"])
    ap.add_argument("--model", type=str, default=None,
                    help="Model name (openai: gpt-4o, ollama: llama3.1:8b, etc.).")
    ap.add_argument("--ollama_url", type=str, default="http://localhost:11434")
    ap.add_argument("--lit_kg_path", type=str, default=None,
                    help="Override path to the Stage 1 KG (default: stage_I/global_knowledge_graph.ttl).")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max_tokens", type=int, default=900)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    # Resolve input: either a prepared file, or run an adapter to build one.
    if args.adapter:
        if not args.adapter_input:
            raise SystemExit("--adapter requires --adapter_input")
        rows = _run_adapter(args.adapter, Path(args.adapter_input))
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        staged = Path(f"/tmp/kv_stage2_input_{args.adapter}_{ts}.jsonl")
        with staged.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
        print(f"Adapter '{args.adapter}' produced {len(rows)} row(s) → {staged}")
        input_path = staged
    elif args.input:
        input_path = Path(args.input)
    else:
        raise SystemExit("Provide --input or (--adapter and --adapter_input).")

    cfg = Stage2Config(repo_root=_KORAL_DIR, temperature=args.temperature, max_tokens=args.max_tokens)
    runner = KVStage2Runner(
        cfg=cfg,
        llm_backend=args.llm_backend,
        llm_model=args.model,
        llm_base_url=args.ollama_url if args.llm_backend == "ollama" else None,
        lit_kg_path=Path(args.lit_kg_path) if args.lit_kg_path else None,
    )

    out_name = args.out_name or f"kv_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    outs = runner.run(
        input_path=input_path,
        tasks=tasks,
        out_name=out_name,
        limit_rows=args.limit_rows,
        seed=args.seed,
    )
    print(f"Run saved under:       {outs.run_dir}")
    print(f"Responses (JSONL):     {outs.responses_jsonl}")
    print(f"Per-sample metrics:    {outs.metrics_csv}")
    print(f"Summary:               {outs.summary_json}")
    print(f"Data KG TTLs:          {outs.data_kg_dir}")


if __name__ == "__main__":
    main()
