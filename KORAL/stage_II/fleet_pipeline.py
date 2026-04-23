#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fleet-level Stage II runner (collective analysis for Table II).

This module provides a fleet-scale analogue to stage_II.pipeline.Stage2Runner.
It groups N drives (or drive-windows) into a cohort and makes ONE LLM call per
task for that cohort, then computes metrics at the drive level.

Supported dataset types for Table II (as used in the paper):
- SMART_ALIBABA
- SMART_GOOGLE
- SMART_WORKLOAD

Input format:
- A CSV produced by the data_preparation scripts where each row represents one drive window
  and SMART columns store 30-day series (JSON list or delimited list).
- Must contain a drive identifier column (disk_id or drive_id) and a label (failure or label).
- Optional: ttf_days column for TTF regression (recommended).
- For SMART_WORKLOAD: include app column (Alibaba workload tag).

Outputs written under stage_II/runs/<out_name>/:
- cohort_composition.csv
- responses_fleet.jsonl
- metrics_fleet_per_cohort.csv
- metrics_summary_fleet.json
- fleet_kg_ttl/<cohort_id>.ttl   (if rdflib is installed)
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from stage_II.config import Stage2Config, resolve_path
from stage_II.features.smart import build_smart_ir, infer_smart_columns
from stage_II.features.workload import build_workload_ir
from stage_II.kg.fleet_data_kg import build_fleet_data_kg
from stage_II.kg.literature_kg import LiteratureKG
from stage_II.llm.openai_client import OpenAIChatClient
from stage_II.prompts.templates import system_prompt
from stage_II.prompts.fleet_templates import (
    fleet_predictive_user_prompt,
    fleet_descriptive_user_prompt,
    fleet_prescriptive_user_prompt,
    fleet_whatif_user_prompt,
)
from stage_II.utils.io import ensure_dir, read_csv, write_json, append_jsonl, write_csv
from stage_II.utils.json_utils import extract_json_object
from stage_II.evaluation.metrics_predictive import confusion_from_labels, mse
from stage_II.evaluation.metrics_text import bleu4, rouge_l_f1
from stage_II.evaluation.grounding import faithfulness_precision, counterfactual_validity

def _safe_id(s: str, max_len: int = 72) -> str:
    s = str(s).strip().replace(" ", "_")
    s = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)
    return (s or "UNKNOWN")[:max_len]

def _infer_drive_id(row: Dict[str, Any]) -> Optional[str]:
    for k in ["disk_id", "drive_id", "drive", "id"]:
        if k in row and row.get(k) is not None and str(row.get(k)).strip() != "":
            return str(row.get(k))
    return None

def _risk_score_from_smart_ir(ir: Dict[str, Any]) -> float:
    """Heuristic risk score from SMART IR: slope magnitude + outliers + changepoints."""
    score = 0.0
    frames = ir.get("smart", [])
    if not isinstance(frames, list):
        return 0.0
    for af in frames:
        if not isinstance(af, dict):
            continue
        slope = af.get("slope")
        outl = af.get("outliers", 0) or 0
        cp = af.get("changepoint_idx")
        if slope is not None:
            try:
                score += min(10.0, abs(float(slope)) * 1.5)
            except Exception:
                pass
        try:
            score += min(10.0, float(outl) * 0.75)
        except Exception:
            pass
        if cp is not None:
            score += 1.0
    return float(score)

def _top_signals(ir: Dict[str, Any], drive_id: str, k: int = 2) -> List[Dict[str, Any]]:
    frames = ir.get("smart", [])
    if not isinstance(frames, list):
        return []
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for af in frames:
        if not isinstance(af, dict):
            continue
        s = 0.0
        if af.get("slope") is not None:
            try:
                s += abs(float(af.get("slope"))) * 1.5
            except Exception:
                pass
        try:
            s += float(af.get("outliers", 0) or 0) * 0.75
        except Exception:
            pass
        if af.get("changepoint_idx") is not None:
            s += 1.0
        scored.append((s, af))
    scored.sort(key=lambda x: x[0], reverse=True)

    did = _safe_id(drive_id)
    out = []
    for _, af in scored[:k]:
        attr = str(af.get("attribute") or af.get("id") or "UNK")
        rid = f"DRV_{did}_AF_{attr}"
        out.append({
            "id": rid,
            "attribute": attr,
            "median": af.get("median"),
            "p95": af.get("p95"),
            "min": af.get("min"),
            "max": af.get("max"),
            "slope": af.get("slope"),
            "changepoint_idx": af.get("changepoint_idx"),
            "outliers": af.get("outliers"),
            "n": af.get("n"),
        })
    return out

def _pct(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    idx = int(round((p / 100.0) * (len(s) - 1)))
    idx = max(0, min(idx, len(s) - 1))
    return float(s[idx])

def _fleet_aggregate_attributes(per_drive_ir: List[Tuple[str, Dict[str, Any]]], smart_cols: List[str]) -> List[Dict[str, Any]]:
    """Aggregate per-drive SMART summaries into fleet-level frames."""
    agg = []
    for attr in smart_cols:
        medians: List[float] = []
        slopes: List[float] = []
        outliers: List[float] = []
        cps = 0
        n_have = 0
        top_anom: List[Tuple[float, str]] = []

        for drive_id, ir in per_drive_ir:
            frames = ir.get("smart", [])
            if not isinstance(frames, list):
                continue
            hit = None
            for af in frames:
                if isinstance(af, dict) and af.get("attribute") == attr:
                    hit = af
                    break
            if hit is None:
                continue
            n_have += 1
            if hit.get("median") is not None:
                try: medians.append(float(hit["median"]))
                except Exception: pass
            if hit.get("slope") is not None:
                try: slopes.append(float(hit["slope"]))
                except Exception: pass
            try:
                outliers.append(float(hit.get("outliers", 0) or 0))
            except Exception:
                pass
            if hit.get("changepoint_idx") is not None:
                cps += 1

            # anomaly score for this drive+attr
            s = 0.0
            if hit.get("slope") is not None:
                try: s += abs(float(hit["slope"])) * 1.5
                except Exception: pass
            try: s += float(hit.get("outliers", 0) or 0) * 0.75
            except Exception: pass
            if hit.get("changepoint_idx") is not None:
                s += 1.0
            top_anom.append((s, drive_id))

        if n_have == 0:
            continue

        top_anom.sort(key=lambda x: x[0], reverse=True)
        top_ids = [d for _, d in top_anom[:5]]

        frame = {
            "id": f"FLEET_AF_{attr}",
            "attribute": attr,
            "n_drives": int(n_have),
            "median_of_medians": float(sorted(medians)[len(medians)//2]) if medians else None,
            "p95_of_medians": _pct(medians, 95) if medians else None,
            "mean_slope": float(sum(slopes)/len(slopes)) if slopes else None,
            "p95_abs_slope": _pct([abs(x) for x in slopes], 95) if slopes else None,
            "mean_outliers": float(sum(outliers)/len(outliers)) if outliers else None,
            "frac_changepoint": float(cps / n_have) if n_have else 0.0,
            "top_anomalous_drives": top_ids,
        }
        agg.append(frame)
    return agg

def _workload_distribution(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    counts: Dict[str, int] = {}
    for r in rows:
        a = r.get("app")
        if a is None:
            continue
        s = str(a).strip()
        if not s:
            continue
        counts[s] = counts.get(s, 0) + 1
    if not counts:
        return None
    items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    out: Dict[str, Any] = {"id": "WL_DIST", "type": "app_histogram", "num_unique_apps": int(len(counts))}
    for k, v in items:
        out[f"app_{_safe_id(k, 24)}"] = int(v)
    return out

def _infer_query_terms(dataset_type: str, has_workload: bool) -> List[str]:
    terms = ["SMART", "SSD", "fleet", "cohort", "wear", "uncorrectable", "ECC"]
    if has_workload:
        terms.append("workload")
    if "GOOGLE" in dataset_type:
        terms.append("google")
    if "ALIBABA" in dataset_type:
        terms.append("alibaba")
    return terms

def _default_fleet_whatif(dataset_type: str) -> str:
    if "WORKLOAD" in dataset_type:
        return "If the workload shifts to higher write intensity (rwmixread decreases by 20 points), how do wear-related SMART signals and fleet failure risk change?"
    return "If inlet temperature decreases by 5°C and humidity decreases by 10%, how do tail latency and failure risk change across the fleet?"

@dataclass
class FleetRunOutputs:
    run_dir: Path
    responses_jsonl: Path
    metrics_csv: Path
    summary_json: Path
    fleet_kg_dir: Path

class FleetStage2Runner:
    """Fleet-level runner for Table II style evaluation."""

    def __init__(self, cfg: Stage2Config):
        self.cfg = cfg
        self.repo_root = cfg.repo_root

        # System prompt
        base_cot_path = resolve_path(self.repo_root, Path("ssd_cot_prompt.txt"))
        try:
            base_cot = Path(base_cot_path).read_text(encoding="utf-8")
        except Exception:
            base_cot = ""
        self.sys = system_prompt(base_cot)

        # LLM
        self.llm = OpenAIChatClient(model=cfg.model)

        # Literature KG
        self.lit = LiteratureKG(resolve_path(self.repo_root, cfg.global_kg_ttl_path))
        try:
            self.lit.load()
        except Exception:
            pass

    def run(
        self,
        dataset_type: str,
        input_csv: Path,
        tasks: List[str],
        out_name: str,
        cohort_size: int = 100,
        num_cohorts: int = 1,
        seed: int = 7,
        top_k_signals: int = 2,
        limit_rows: Optional[int] = None,
        refs_csv: Optional[Path] = None,
    ) -> FleetRunOutputs:
        """Run fleet-level evaluation.

        refs_csv (optional): cohort-level curated references for BLEU/ROUGE.
          Required columns: cohort_id, ref_descriptive, ref_prescriptive, ref_whatif
        """
        df = read_csv(input_csv)
        if limit_rows is not None:
            df = df.head(int(limit_rows)).copy()

        smart_cols = infer_smart_columns(list(df.columns))
        has_workload = "app" in df.columns or "fio_job" in df.columns or "workload" in df.columns

        # Row dicts
        rows = [r.to_dict() for _, r in df.iterrows()]
        for r in rows:
            if "drive_id" not in r or r.get("drive_id") is None:
                did = _infer_drive_id(r)
                if did is not None:
                    r["drive_id"] = did

        # Deduplicate by drive_id if possible (table II uses drives)
        uniq: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            did = r.get("drive_id")
            if did is None:
                continue
            if did not in uniq:
                uniq[did] = r
        if len(uniq) >= max(1, cohort_size):
            rows = list(uniq.values())

        # Deterministic shuffle + cohorting
        import random
        rnd = random.Random(seed)
        rnd.shuffle(rows)

        cohorts: List[Tuple[str, List[Dict[str, Any]]]] = []
        idx = 0
        for c in range(int(num_cohorts)):
            if idx + cohort_size > len(rows):
                rnd.shuffle(rows)
                idx = 0
            cohort = rows[idx:idx + cohort_size]
            idx += cohort_size
            cohorts.append((f"cohort_{c}", cohort))

        run_dir = ensure_dir(resolve_path(self.repo_root, self.cfg.runs_dir) / out_name)
        fleet_kg_dir = ensure_dir(run_dir / "fleet_kg_ttl")
        responses_jsonl = run_dir / "responses_fleet.jsonl"
        metrics_csv = run_dir / "metrics_fleet_per_cohort.csv"
        summary_json = run_dir / "metrics_summary_fleet.json"

        # Save cohort composition
        comp = []
        for cid, cohort in cohorts:
            for r in cohort:
                comp.append({
                    "cohort_id": cid,
                    "drive_id": r.get("drive_id"),
                    "disk_id": r.get("disk_id"),
                    "ds": r.get("ds") or r.get("date"),
                    "app": r.get("app") if has_workload else None,
                    "failure": r.get("failure") if "failure" in r else r.get("label"),
                    "ttf_days": r.get("ttf_days") if "ttf_days" in r else r.get("ttf"),
                })
        write_csv(run_dir / "cohort_composition.csv", pd.DataFrame(comp))

        # Optional references
        ref_lookup: Dict[str, Dict[str, str]] = {}
        if refs_csv is not None and Path(refs_csv).exists():
            rdf = pd.read_csv(refs_csv)
            for _, rr in rdf.iterrows():
                cid = str(rr.get("cohort_id"))
                ref_lookup[cid] = {
                    "ref_descriptive": str(rr.get("ref_descriptive", "")) if rr.get("ref_descriptive") is not None else "",
                    "ref_prescriptive": str(rr.get("ref_prescriptive", "")) if rr.get("ref_prescriptive") is not None else "",
                    "ref_whatif": str(rr.get("ref_whatif", "")) if rr.get("ref_whatif") is not None else "",
                }

        rows_out = []
        metric_rows = []

        y_true_all: List[int] = []
        y_pred_all: List[int] = []
        ttf_true_all: List[float] = []
        ttf_pred_all: List[float] = []

        # Text metric accumulators
        b4_desc, rl_desc, fip_desc = [], [], []
        b4_pres, rl_pres, fip_pres = [], [], []
        b4_wif, rl_wif, cfv_wif = [], [], []

        rng = int(seed)

        for cid, cohort in cohorts:
            per_drive_ir: List[Tuple[str, Dict[str, Any]]] = []
            drive_summaries: List[Dict[str, Any]] = []

            for r in cohort:
                drive_id = str(r.get("drive_id") or _infer_drive_id(r) or "UNKNOWN")
                ir: Dict[str, Any] = {}
                ir.update(build_smart_ir(r, smart_cols))
                if has_workload:
                    ir.update(build_workload_ir(r))
                per_drive_ir.append((drive_id, ir))

                drive_summaries.append({
                    "drive_id": drive_id,
                    "ds": r.get("ds") or r.get("date"),
                    "app": r.get("app") if has_workload else None,
                    "risk_score": _risk_score_from_smart_ir(ir),
                    "top_signals": _top_signals(ir, drive_id, k=top_k_signals),
                })

            # Fleet aggregates from ALL drives
            fleet_aggs = _fleet_aggregate_attributes(per_drive_ir, smart_cols)
            wd = _workload_distribution(cohort) if has_workload else None

            fleet_payload = {
                "fleet": {
                    "cohort_id": cid,
                    "dataset_type": dataset_type,
                    "cohort_size": int(len(cohort)),
                    "window_days": int(self.cfg.fail_horizon_days),
                    "notes": "Fleet payload contains per-drive compact summaries + fleet aggregates.",
                    "aggregate_attributes": fleet_aggs,
                    "workload_distribution": wd,
                    "drives": drive_summaries,
                }
            }

            # Fleet DataKG
            kg_art = build_fleet_data_kg(cid, fleet_payload)
            if kg_art.ttl:
                (fleet_kg_dir / f"{cid}.ttl").write_text(kg_art.ttl, encoding="utf-8")

            # Literature retrieval
            terms = _infer_query_terms(dataset_type, has_workload)
            lit_evidence = self.lit.retrieve(terms, limit=10)
            lit_payload = [{"id": e.id, "text": e.text, "source": e.source} for e in lit_evidence]

            prompt_payload = {
                "cohort_id": cid,
                "dataset_type": dataset_type,
                "cohort_size": int(len(cohort)),
                "FleetIR": fleet_payload,
                "DataKG_refs": sorted(list(kg_art.refs)),
                "Literature": lit_payload,
                "instructions": {
                    "positive_predictions_only": "For predictive, list only predicted failing drives.",
                    "evidence": "Cite ref IDs (DataKG or literature) for claims.",
                },
            }

            available_refs: Set[str] = set(kg_art.refs)
            for e in lit_evidence:
                available_refs.add(e.id)

            # Run tasks
            for task in tasks:
                if task == "predictive":
                    user = fleet_predictive_user_prompt(prompt_payload)
                elif task == "descriptive":
                    user = fleet_descriptive_user_prompt(prompt_payload)
                elif task == "prescriptive":
                    user = fleet_prescriptive_user_prompt(prompt_payload)
                elif task == "whatif":
                    scenario = _default_fleet_whatif(dataset_type)
                    user = fleet_whatif_user_prompt(prompt_payload, scenario)
                else:
                    raise ValueError(f"Unknown task: {task}")

                resp = self.llm.chat(
                    system=self.sys,
                    user=user,
                    temperature=self.cfg.temperature,
                    max_tokens=self.cfg.max_tokens,
                    seed=rng,
                )
                rng += 1

                parsed = extract_json_object(resp.text) or {"task": task, "cohort_id": cid, "parse_error": True, "raw_text": resp.text}

                rows_out.append({
                    "cohort_id": cid,
                    "task": task,
                    "prompt_terms": terms,
                    "response_text": resp.text,
                    "response_json": parsed,
                })

                m: Dict[str, Any] = {"cohort_id": cid, "task": task}

                # Grounding metrics
                if task in ("descriptive", "prescriptive"):
                    m["FiP"] = faithfulness_precision(parsed, available_refs)
                if task == "whatif":
                    m["CFV"] = counterfactual_validity(parsed, direction_lookup=None)
                metric_rows.append(m)

                # Predictive: map list of predicted drives → per-drive predictions
                if task == "predictive":
                    pred_list = parsed.get("predicted_failing_drives", [])
                    pred_ids: Set[str] = set()
                    pred_ttf: Dict[str, float] = {}

                    if isinstance(pred_list, list):
                        for it in pred_list:
                            if isinstance(it, str):
                                pred_ids.add(it.strip())
                            elif isinstance(it, dict):
                                did = it.get("drive_id") or it.get("disk_id")
                                if did:
                                    dids = str(did).strip()
                                    pred_ids.add(dids)
                                    v = it.get("predicted_ttf_days", None)
                                    if v is not None:
                                        try:
                                            pred_ttf[dids] = float(v)
                                        except Exception:
                                            pass

                    for r in cohort:
                        did = str(r.get("drive_id") or _infer_drive_id(r) or "UNKNOWN")
                        gt = r.get("failure", None)
                        if gt is None:
                            gt = r.get("label", None)
                        if gt is None:
                            continue
                        y_true_all.append(int(gt))
                        y_pred_all.append(1 if did in pred_ids else 0)

                        gt_ttf = r.get("ttf_days", None)
                        if gt_ttf is None:
                            gt_ttf = r.get("ttf", None)
                        if gt_ttf is not None and did in pred_ttf:
                            try:
                                ttf_true_all.append(float(gt_ttf))
                                ttf_pred_all.append(float(pred_ttf[did]))
                            except Exception:
                                pass

                # Text overlap metrics if references are provided
                if task == "descriptive":
                    ref = ref_lookup.get(cid, {}).get("ref_descriptive", "")
                    gen = parsed.get("summary") or parsed.get("rationale") or ""
                    if ref and gen:
                        b4_desc.append(bleu4(str(gen), str(ref)))
                        rl_desc.append(rouge_l_f1(str(gen), str(ref)))
                    fip_desc.append(float(m.get("FiP", 0.0)))
                if task == "prescriptive":
                    ref = ref_lookup.get(cid, {}).get("ref_prescriptive", "")
                    gen = parsed.get("recommendations") or ""
                    if ref and gen:
                        b4_pres.append(bleu4(str(gen), str(ref)))
                        rl_pres.append(rouge_l_f1(str(gen), str(ref)))
                    fip_pres.append(float(m.get("FiP", 0.0)))
                if task == "whatif":
                    ref = ref_lookup.get(cid, {}).get("ref_whatif", "")
                    gen = parsed.get("analysis") or ""
                    if ref and gen:
                        b4_wif.append(bleu4(str(gen), str(ref)))
                        rl_wif.append(rouge_l_f1(str(gen), str(ref)))
                    cfv_wif.append(float(m.get("CFV", 0.0)))

                time.sleep(0.2)

        # Persist outputs
        append_jsonl(responses_jsonl, rows_out)
        write_csv(metrics_csv, pd.DataFrame(metric_rows))

        # Aggregate summary (Table II-style)
        summary: Dict[str, Any] = {
            "dataset_type": dataset_type,
            "cohort_size": int(cohort_size),
            "num_cohorts": int(num_cohorts),
        }

        # Predictive metrics across all drives
        pred_metrics: Dict[str, Any] = {}
        if y_true_all and y_pred_all:
            conf = confusion_from_labels(y_true_all, y_pred_all)
            pred_metrics = {
                "P": conf.precision(),
                "R": conf.recall(),
                "A": conf.accuracy(),
                "TP": conf.tp, "FP": conf.fp, "FN": conf.fn, "TN": conf.tn,
            }
        if ttf_true_all and ttf_pred_all:
            pred_metrics["TTF_MSE"] = mse(ttf_true_all, ttf_pred_all)
        summary["predictive"] = pred_metrics

        # Descriptive / Prescriptive / What-if aggregates
        def _avg(x: List[float]) -> Optional[float]:
            return float(sum(x)/len(x)) if x else None

        summary["descriptive"] = {
            "B4": _avg(b4_desc),
            "RL": _avg(rl_desc),
            "FiP": _avg(fip_desc),
            "n_ref": int(len(b4_desc)),
        }
        summary["prescriptive"] = {
            "B4": _avg(b4_pres),
            "RL": _avg(rl_pres),
            "FiP": _avg(fip_pres),
            "n_ref": int(len(b4_pres)),
        }
        summary["whatif"] = {
            "B4": _avg(b4_wif),
            "RL": _avg(rl_wif),
            "CFV": _avg(cfv_wif),
            "n_ref": int(len(b4_wif)),
        }
        summary["notes"] = {
            "refs_csv": str(refs_csv) if refs_csv is not None else None,
            "if_refs_missing": "BLEU/ROUGE are only computed when refs_csv provides references; grounding metrics (FiP/CFV) are always computed.",
        }

        write_json(summary_json, summary)

        return FleetRunOutputs(
            run_dir=run_dir,
            responses_jsonl=responses_jsonl,
            metrics_csv=metrics_csv,
            summary_json=summary_json,
            fleet_kg_dir=fleet_kg_dir,
        )
