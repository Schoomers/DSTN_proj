#!/usr/bin/env bash
# End-to-end demo for the KV-SSD diagnosis project.
# Narrate along; each section pauses for ENTER so you control the pace.

set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"

# Colors for readability
B=$'\033[1m'; C=$'\033[36m'; Y=$'\033[33m'; G=$'\033[32m'; R=$'\033[0m'
pause() { printf "\n${Y}── press ENTER to continue ──${R}\n"; read -r; }
hdr()   { printf "\n${B}${C}═══ %s ═══${R}\n" "$*"; }

TRACE_DIR="/Users/sahilmaheshwari/Desktop/Code/DSTN_Groupproj/test/cache-trace/samples/2020Mar"
CLUSTER_A=cluster018    # read-heavy, small values
CLUSTER_B=cluster012    # write-heavy, bimodal values
MODEL="${MODEL:-llama3.2:3b}"

# -----------------------------------------------------------
hdr "1. Venv + Ollama check"
source venv/bin/activate
printf "python: %s\n" "$(which python)"
if ! curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null; then
  echo "${Y}Ollama not responding — start it with:${R}"
  echo "  /opt/homebrew/opt/ollama/bin/ollama serve &"
  exit 1
fi
ollama list | grep -F "$MODEL" || { echo "Model $MODEL not pulled"; exit 1; }
echo "${G}Ollama ready, model $MODEL available.${R}"
pause

# -----------------------------------------------------------
hdr "2. Stage 1 inputs: papers + prompt + taxonomy"
ls dataset/ | head -12
echo
echo "${C}── first 30 lines of the Stage 1 prompt ──${R}"
sed -n '1,30p' prompts/ssd_cot_prompt.txt
pause

# -----------------------------------------------------------
hdr "3. Stage 1 output: global KG (pre-computed)"
echo "Global KG size:"
wc -l stage_I/global_knowledge_graph.ttl
echo
echo "KV-SSD paper entities we extracted (LKV=LearnedKV, KVS=KVSSD, DOT=Dotori):"
printf "  LKV: %s entities\n" "$(grep -c 'ns1:LKV_E' stage_I/global_knowledge_graph.ttl || true)"
printf "  KVS: %s entities\n" "$(grep -c 'ns1:KVS_E' stage_I/global_knowledge_graph.ttl || true)"
printf "  DOT: %s entities\n" "$(grep -c 'ns1:DOT_E' stage_I/global_knowledge_graph.ttl || true)"
echo
echo "${C}── example entity from the KG ──${R}"
awk '/ns1:LKV_E1_LearnedKV a/,/^$/' stage_I/global_knowledge_graph.ttl | head -12
pause

# -----------------------------------------------------------
hdr "4. Stage 1 KG retrieval demo (what Stage 2 will use)"
python - <<'PY'
from stage_II.kg.literature_kg import LiteratureKG
from pathlib import Path
lit = LiteratureKG(Path("stage_I/global_knowledge_graph.ttl")); lit.load()
print("Retrieval for terms: CMT, Inline Ratio, Translation Page, Mapping Entry Churn")
print("-" * 70)
for e in lit.retrieve(["CMT", "Inline Ratio", "Translation Page", "Mapping Entry Churn"], limit=6):
    print(f"{e.id:8s} :: {e.text[:110]}")
PY
pause

# -----------------------------------------------------------
hdr "5. Stage 2 input: Twitter production trace"
echo "File: $TRACE_DIR/$CLUSTER_A"
wc -l "$TRACE_DIR/$CLUSTER_A"
echo
echo "${C}── first 3 rows (timestamp, key, key_size, value_size, client_id, op, TTL) ──${R}"
head -3 "$TRACE_DIR/$CLUSTER_A"
echo
echo "${C}── stat row from the OSDI'20 paper ──${R}"
grep -F "| cluster18 " stat/2020Mar.md 2>/dev/null || \
  grep -F "| cluster18 " /Users/sahilmaheshwari/Desktop/Code/DSTN_Groupproj/test/cache-trace/stat/2020Mar.md | head -1
pause

# -----------------------------------------------------------
hdr "6. LIVE Stage 2 run: cluster018 (read-heavy, small values)"
echo "Expect: 1M-row trace parse (<1s) + 4 LLM calls (~20-60s each)"
echo
python -m stage_II.kv_cli \
  --adapter twitter \
  --adapter_input "$TRACE_DIR/$CLUSTER_A" \
  --llm_backend ollama \
  --model "$MODEL" \
  --out_name demo_cluster018
pause

# -----------------------------------------------------------
hdr "7. Inspect the LLM outputs"
python - <<'PY'
import json, pathlib
p = pathlib.Path("stage_II/runs/demo_cluster018/responses.jsonl")
for line in p.read_text().splitlines():
    r = json.loads(line); rj = r["response_json"]
    print(f"\n━━━ {r['task'].upper()} ━━━")
    if r["task"] == "diagnose":
        print("SUMMARY:", (rj.get("summary","") or "")[:300])
        for m in rj.get("likely_mechanisms", [])[:2]:
            print(f"  ► {m.get('mechanism')} :: support={m.get('support')}")
    elif r["task"] == "attribute":
        pc = rj.get("primary_cause", {}) or {}
        print(f"PRIMARY CAUSE: {pc.get('mechanism')} (conf={pc.get('confidence')})")
        print(f"  explanation: {(pc.get('explanation','') or '')[:200]}")
    elif r["task"] == "prescribe":
        for rec in (rj.get("recommendations", []) or [])[:3]:
            print(f"  ► [{rec.get('priority')}] {rec.get('action','')[:100]}")
            print(f"     target={rec.get('target_signal')} support={rec.get('support')}")
    elif r["task"] == "whatif":
        for s in (rj.get("counterfactual_statements", []) or [])[:2]:
            print(f"  ► {(s.get('statement','') or '')[:140]}")
            print(f"     dir={s.get('effect_direction')} ev={s.get('evidence')}")
PY
echo
echo "${C}── grounding scores ──${R}"
cat stage_II/runs/demo_cluster018/metrics_summary.json
pause

# -----------------------------------------------------------
hdr "8. Data KG artifact (per-sample TTL)"
head -30 stage_II/runs/demo_cluster018/data_kg_ttl/twitter_cluster018.ttl
pause

# -----------------------------------------------------------
hdr "9. OPTIONAL: contrast with cluster012 (write-heavy, bimodal)"
read -r -p "Run cluster012 too? [y/N] " ans
if [[ "$ans" == "y" || "$ans" == "Y" ]]; then
  python -m stage_II.kv_cli \
    --adapter twitter \
    --adapter_input "$TRACE_DIR/$CLUSTER_B" \
    --llm_backend ollama --model "$MODEL" \
    --out_name demo_cluster012
  echo
  echo "${C}── cluster012 diagnose SUMMARY ──${R}"
  python - <<'PY'
import json
for line in open("stage_II/runs/demo_cluster012/responses.jsonl"):
    r = json.loads(line)
    if r["task"] == "diagnose":
        print(r["response_json"].get("summary",""))
        print("likely_mechanisms:", [m.get("mechanism") for m in r["response_json"].get("likely_mechanisms",[])])
        break
PY
fi
pause

# -----------------------------------------------------------
hdr "10. Recap"
cat <<'END'
  ● Stage 1: 4 KV-SSD papers + taxonomy → 388 new triples in the global KG
             (LearnedKV + KVSSD + Dotori entities, all with paper provenance)
  ● Stage 2: 1M real Twitter KV requests → 4 grounded diagnoses
             FiP = 1.0 / CFV = 1.0 on a 3B-parameter local model
  ● Next:    MoKE+KVPack integration for real CMT / inline-ratio internals
             (currently estimated under a KVPack frame threshold)
END
END
echo "${G}demo complete.${R}"
