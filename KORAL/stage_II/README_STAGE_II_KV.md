# KORAL Stage II — KV-SSD Diagnosis Pipeline

This is the KV-SSD path of Stage II. It sits next to the existing block-SSD/SMART pipeline (`cli.py`, `fleet_cli.py`) and shares the same utilities, grounding, and metric evaluators. It does NOT modify the block-SSD pipeline.

## What it does

```
  KV telemetry sample
        │
        ▼
  features/kv.py  ──────►  KV IR  (inline_ratio, cmt_hit_rate, mapping_churn, …)
        │
        ▼
  kg/data_kg_kv.py  ───►  per-sample KV DataKG TTL  (maps frames to SSD/KVSSD/* classes)
        │
        ▼
  kg/literature_kg.py  ►  retrieves evidence from stage_I/global_knowledge_graph.ttl
        │                (KV-SSD-aware terms: CMT, Inline Ratio, Translation Page, …)
        ▼
  LLM  (openai | ollama | none)
        │
        ▼
  metrics (FiP / CFV / B4 / RL)  +  run artifacts
```

## Input schema

A row = one KV telemetry window. Time-series fields accept a JSON list, a comma/semicolon-separated list, or a scalar. Known fields:

| Group | Field |
|---|---|
| identity | `sample_id`, `vendor`, `model`, `firmware`, `capacity_gb`, `interface`, `kv_ftl` |
| workload | `key_size_bytes`, `value_size_bytes`, `value_size_dist`, `read_write_ratio`, `update_intensity`, `access_skew` |
| CMT / mapping | `cmt_hit_rate`, `cmt_eviction_rate`, `cmt_size_entries`, `mapping_entry_churn`, `translation_page_pressure`, `inline_ratio`, `inline_to_regular_rate` |
| performance | `throughput_kops`, `read_latency_p50_us`, `read_latency_p99_us`, `tail_latency_p999_us`, `write_amplification`, `read_amplification`, `gc_overhead_pct` |
| references (optional) | `ref_diagnose`, `ref_attribute`, `ref_prescribe`, `ref_whatif` for text-overlap scoring |

Any unknown column is ignored.

## Adapters

Produce rows in the schema above from common sources:

- `stage_II.adapters.mqsim.parse_mqsim_xml(path)` — MQSim XML results (throughput, latency percentiles, WA/RA, GC overhead).
- `stage_II.adapters.rocksdb_stats.parse_rocksdb_log(path)` — RocksDB LOG or db_bench stdout (ops/sec, p50/p99/p999, WA from cumulative bytes).
- `stage_II.adapters.kvpack.parse_kvpack_jsonl(path)` — KVPack-style simulator JSON/JSONL with native KV-FTL signals.

## Running

From repo root (so `stage_II` is importable as a package) — typically `~/Desktop/Code/DSTN_Groupproj/KORAL`:

**Dry run (no LLM) — validates plumbing only:**
```bash
python -m stage_II.kv_cli \
  --input stage_II/examples/kv_sample.jsonl \
  --llm_backend none \
  --out_name kv_dryrun
```

**Local Ollama (offline; use same model as Stage 1):**
```bash
python -m stage_II.kv_cli \
  --input stage_II/examples/kv_sample.jsonl \
  --llm_backend ollama \
  --model llama3.1:8b \
  --out_name kv_ollama_demo
```

**OpenAI (requires `OPENAI_API_KEY`):**
```bash
export OPENAI_API_KEY=sk-...
python -m stage_II.kv_cli \
  --input stage_II/examples/kv_sample.jsonl \
  --llm_backend openai \
  --model gpt-4o \
  --out_name kv_gpt4o_demo
```

**From a MQSim run:**
```bash
python -m stage_II.kv_cli \
  --adapter mqsim --adapter_input ../MQSim/out_dir \
  --llm_backend ollama --model llama3.1:8b \
  --out_name kv_mqsim_run1
```

## Outputs

```
stage_II/runs/<out_name>/
  input_samples.jsonl          # echo of what the runner actually saw
  data_kg_ttl/<sample_id>.ttl  # per-sample KV DataKG
  responses.jsonl              # full LLM I/O + parsed JSON per (sample, task)
  metrics_per_sample.csv       # FiP / CFV / B4 / RL per row
  metrics_summary.json         # per-task aggregates
```

## Tasks

- **diagnose** — descriptive narrative citing KV-FTL mechanisms from the retrieved KG.
- **attribute** — ranks candidate KV-FTL causes (CMT pressure, inline→regular, churn, etc.).
- **prescribe** — tuning recommendations (inline threshold, CMT size, GC trigger, KV-separation policy).
- **whatif** — counterfactual reasoning with direction-consistent evidence.

Select a subset with `--tasks diagnose,attribute`.

## Grounding

`FiP` (Faithfulness-in-Prompt) = atomic claims whose `support` IDs all resolve to a known IR or LIT reference. `CFV` (Counterfactual Validity) = what-if statements that cite evidence and state a direction. Both implemented in `evaluation/grounding.py` (shared with the block-SSD path).
