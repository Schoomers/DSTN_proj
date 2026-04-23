#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage II KV-SSD diagnosis pipeline.

End-to-end:
  telemetry sample  →  KV IR (features/kv.py)
                    →  KV DataKG TTL (kg/data_kg_kv.py)
                    →  Stage 1 KG retrieval (kg/literature_kg.py, SSD/KVSSD/* classes)
                    →  LLM (OpenAI | Ollama | none)
                    →  grounding / text / predictive metrics
"""

from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # graceful fallback
    def tqdm(it, **_):
        return it

from stage_II.config import Stage2Config, resolve_path
from stage_II.features.kv import (
    build_kv_ir,
    collect_kv_refs,
    kv_observations,
    kv_query_terms,
)
from stage_II.kg.data_kg_kv import build_kv_data_kg
from stage_II.kg.literature_kg import LiteratureKG
from stage_II.prompts.kv_templates import (
    kv_system_prompt,
    diagnose_user_prompt,
    attribute_user_prompt,
    prescribe_user_prompt,
    whatif_user_prompt,
)
from stage_II.utils.io import ensure_dir, read_csv, write_json, append_jsonl, write_csv
from stage_II.utils.json_utils import extract_json_object
from stage_II.evaluation.metrics_text import bleu4, rouge_l_f1
from stage_II.evaluation.grounding import faithfulness_precision, counterfactual_validity


# ---------- Input handling ----------

def _read_input(path: Path) -> List[Dict[str, Any]]:
    """Accept CSV or JSONL; return list of row dicts."""
    p = Path(path)
    if p.suffix.lower() in (".jsonl", ".ndjson"):
        rows: List[Dict[str, Any]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
        return rows
    if p.suffix.lower() == ".json":
        obj = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(obj, dict):
            return [obj]
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)]
        raise ValueError(f"Unsupported JSON shape in {p}")
    df = read_csv(p)
    return df.to_dict(orient="records")


def _default_whatif(ir: Dict[str, Any]) -> str:
    scalar = (ir.get("kv") or {}).get("scalar") or {}
    if "cmt_hit_rate" in scalar:
        return "If CMT size is doubled (raising cmt_hit_rate by ~10 points), how do tail latency and read amplification respond?"
    if "inline_ratio" in scalar:
        return "If the inline threshold is raised to keep ~80% of entries inline, how do mapping-entry churn and p99.9 read latency change?"
    return "If value sizes shift upward by 2x (pushing inline→regular transitions), what happens to CMT pressure and tail latency?"


# ---------- LLM backends ----------

class _NoLLM:
    """Skip LLM calls; return a placeholder response."""
    def chat(self, system: str, user: str, **_) -> "LLMResponse":
        stub = {"task": "dry_run", "note": "llm_backend=none; no call performed"}
        return LLMResponse(text=json.dumps(stub), raw={})


@dataclass
class LLMResponse:
    text: str
    raw: Dict[str, Any]


def _make_llm(backend: str, model: str, base_url: Optional[str]):
    backend = (backend or "openai").lower()
    if backend == "none":
        return _NoLLM()
    if backend == "ollama":
        import requests
        class _Ollama:
            def __init__(self, m, url):
                self.model = m
                self.url = (url or "http://localhost:11434").rstrip("/")
            def chat(self, system: str, user: str, temperature: float = 0.2,
                     max_tokens: int = 900, seed: Optional[int] = None, **_) -> LLMResponse:
                prompt = f"system: {system}\n\nuser: {user}"
                r = requests.post(
                    f"{self.url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json",
                        "options": {"temperature": float(temperature)},
                    },
                    timeout=180,
                )
                r.raise_for_status()
                data = r.json()
                return LLMResponse(text=data.get("response", ""), raw=data)
        return _Ollama(model, base_url)
    # default: openai
    from stage_II.llm.openai_client import OpenAIChatClient
    return OpenAIChatClient(model=model)


# ---------- Runner ----------

TASKS = ("diagnose", "attribute", "prescribe", "whatif")


@dataclass
class KVRunOutputs:
    run_dir: Path
    responses_jsonl: Path
    metrics_csv: Path
    summary_json: Path
    data_kg_dir: Path


class KVStage2Runner:
    def __init__(
        self,
        cfg: Stage2Config,
        llm_backend: str = "openai",
        llm_model: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        lit_kg_path: Optional[Path] = None,
    ):
        self.cfg = cfg
        self.repo_root = cfg.repo_root
        self.llm = _make_llm(
            llm_backend,
            llm_model or cfg.model,
            llm_base_url,
        )
        # Default literature KG = Stage 1 global KG (already KV-SSD-extended).
        if lit_kg_path is None:
            lit_kg_path = resolve_path(self.repo_root, Path("stage_I/global_knowledge_graph.ttl"))
        self.lit = LiteratureKG(Path(lit_kg_path))
        try:
            self.lit.load()
        except Exception:
            pass  # falls back to grep

    def run(
        self,
        input_path: Path,
        tasks: List[str],
        out_name: str,
        limit_rows: Optional[int] = None,
        seed: int = 7,
    ) -> KVRunOutputs:
        rows = _read_input(Path(input_path))
        if limit_rows is not None:
            rows = rows[: int(limit_rows)]

        run_dir = ensure_dir(resolve_path(self.repo_root, self.cfg.runs_dir) / out_name)
        data_kg_dir = ensure_dir(run_dir / "data_kg_ttl")
        responses_jsonl = run_dir / "responses.jsonl"
        metrics_csv = run_dir / "metrics_per_sample.csv"
        summary_json = run_dir / "metrics_summary.json"

        # Reproducibility snapshot
        with (run_dir / "input_samples.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

        sys_prompt = kv_system_prompt()

        responses_out: List[Dict[str, Any]] = []
        metric_rows: List[Dict[str, Any]] = []

        rng = seed
        total_calls = max(1, len(rows) * len(tasks))
        pbar = tqdm(total=total_calls, desc="KV Stage 2", unit="call", dynamic_ncols=True)
        for idx, row in enumerate(rows):
            sample_id = str(row.get("sample_id") or row.get("window_id") or row.get("id") or f"s{idx}")

            ir = build_kv_ir(row)
            obs = kv_observations(ir)
            ir_refs = set(collect_kv_refs(ir))

            # Data KG
            dk = build_kv_data_kg(sample_id, ir)
            if dk.ttl:
                (data_kg_dir / f"{sample_id}.ttl").write_text(dk.ttl, encoding="utf-8")
            available_refs = set(dk.refs) | ir_refs

            # Literature retrieval (KV-SSD-aware terms)
            terms = kv_query_terms(ir)
            lit_evidence = self.lit.retrieve(terms, limit=10)
            lit_payload = [{"id": e.id, "text": e.text, "source": e.source} for e in lit_evidence]
            for e in lit_evidence:
                available_refs.add(e.id)

            sample_payload = {
                "sample_id": sample_id,
                "meta": {
                    "source": row.get("source"),
                    "label": row.get("label"),
                    "anomaly_kind": row.get("anomaly_kind"),
                },
                "IR": ir,
                "observations": obs,
                "DataKG_refs": sorted(available_refs),
                "Literature": lit_payload,
            }

            for task in tasks:
                pbar.set_postfix_str(f"{sample_id} · {task}", refresh=False)
                if task == "diagnose":
                    user = diagnose_user_prompt(sample_payload)
                elif task == "attribute":
                    user = attribute_user_prompt(sample_payload)
                elif task == "prescribe":
                    user = prescribe_user_prompt(sample_payload)
                elif task == "whatif":
                    scenario = str(row.get("whatif_scenario") or _default_whatif(ir))
                    user = whatif_user_prompt(sample_payload, scenario)
                else:
                    raise ValueError(f"Unknown task: {task}")

                try:
                    resp = self.llm.chat(
                        system=sys_prompt,
                        user=user,
                        temperature=self.cfg.temperature,
                        max_tokens=self.cfg.max_tokens,
                        seed=rng,
                    )
                except Exception as ex:
                    resp = LLMResponse(text=json.dumps({"error": str(ex)}), raw={})
                rng += 1

                parsed = extract_json_object(resp.text) or {
                    "task": task, "sample_id": sample_id,
                    "parse_error": True, "raw_text": resp.text,
                }

                responses_out.append({
                    "sample_id": sample_id,
                    "task": task,
                    "prompt_terms": terms,
                    "response_text": resp.text,
                    "response_json": parsed,
                })

                m: Dict[str, Any] = {"sample_id": sample_id, "task": task}
                if task in ("diagnose", "attribute", "prescribe"):
                    m["FiP"] = faithfulness_precision(parsed, available_refs)
                if task == "whatif":
                    m["CFV"] = counterfactual_validity(parsed)
                # Text overlap if reference text was provided
                ref_key = f"ref_{task}"
                if row.get(ref_key):
                    gen = parsed.get("summary") or parsed.get("analysis") or parsed.get("primary_cause", {}).get("explanation") or ""
                    if gen:
                        m["B4"] = bleu4(str(gen), str(row[ref_key]))
                        m["RL"] = rouge_l_f1(str(gen), str(row[ref_key]))
                metric_rows.append(m)

                pbar.update(1)
                time.sleep(0.1)
        pbar.close()

        append_jsonl(responses_jsonl, responses_out)
        mdf = pd.DataFrame(metric_rows)
        write_csv(metrics_csv, mdf)
        write_json(summary_json, self._aggregate(metric_rows))

        return KVRunOutputs(
            run_dir=run_dir,
            responses_jsonl=responses_jsonl,
            metrics_csv=metrics_csv,
            summary_json=summary_json,
            data_kg_dir=data_kg_dir,
        )

    def _aggregate(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        by_task: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            by_task.setdefault(r["task"], []).append(r)
        for task, rs in by_task.items():
            agg: Dict[str, Any] = {"n": len(rs)}
            for k in ("FiP", "CFV", "B4", "RL"):
                vals = [r[k] for r in rs if k in r and r[k] is not None]
                if vals:
                    agg[k] = float(sum(vals) / len(vals))
            out[task] = agg
        return out
