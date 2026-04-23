#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fleet-level Data KG materialization (collective / cohort graph).

This module is used for Table II-style evaluation where a single LLM query
analyzes a cohort of N drives at once (fleet-level collective analysis).

Design goals:
- Keep the KG compact enough to serialize quickly and to provide stable reference IDs
  that the LLM can cite as evidence.
- Avoid duplicating full 30-day series in the KG; instead store robust summaries.

The output is an optional Turtle TTL string (requires rdflib) + a set of reference IDs.
If rdflib is not available, TTL is None but refs still work for grounding metrics.

Reference ID conventions (these are what the LLM should cite):
- Fleet aggregate frame:     FLEET_AF_<attribute>
- Per-drive attribute frame: DRV_<drive_id>_AF_<attribute>
- Workload distribution:     WL_DIST
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

try:
    import rdflib  # type: ignore
    from rdflib import Graph, Namespace, Literal, RDF
except Exception:  # pragma: no cover
    rdflib = None
    Graph = None

@dataclass
class FleetKGArtifact:
    ttl: Optional[str]
    refs: Set[str]

def _safe_id(s: str, max_len: int = 72) -> str:
    s = str(s).strip().replace(" ", "_")
    s = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in s)
    if not s:
        s = "UNKNOWN"
    return s[:max_len]

def build_fleet_data_kg(cohort_id: str, fleet_ir: Dict[str, Any]) -> FleetKGArtifact:
    """Build a fleet-level KG for one cohort.

    Expected fleet_ir structure (produced by fleet_pipeline):
      {
        "fleet": {
          "cohort_id": ...,
          "dataset_type": ...,
          "cohort_size": ...,
          "window_days": ...,
          "aggregate_attributes": [ {"id": "FLEET_AF_r_5", "attribute": "r_5", ...}, ... ],
          "workload_distribution": {"id": "WL_DIST", ...} | None,
          "drives": [
            {
              "drive_id": "...",
              "ds": "...",
              "app": "..." | None,
              "risk_score": <float>,
              "top_signals": [
                {"id": "DRV_<id>_AF_r_5", "attribute": "r_5", "median": ..., ...},
                ...
              ]
            },
            ...
          ]
        }
      }
    """
    refs: Set[str] = set()

    fleet = fleet_ir.get("fleet", {}) if isinstance(fleet_ir, dict) else {}

    # No rdflib → only refs
    if rdflib is None:
        for a in (fleet.get("aggregate_attributes") or []):
            if isinstance(a, dict) and a.get("id"):
                refs.add(str(a["id"]))
        wd = fleet.get("workload_distribution")
        if isinstance(wd, dict) and wd.get("id"):
            refs.add(str(wd["id"]))
        for d in (fleet.get("drives") or []):
            if not isinstance(d, dict):
                continue
            for s in (d.get("top_signals") or []):
                if isinstance(s, dict) and s.get("id"):
                    refs.add(str(s["id"]))
        return FleetKGArtifact(ttl=None, refs=set(r for r in refs if r))

    EX = Namespace("http://example.org/koral-fleet#")
    g = Graph()
    g.bind("ex", EX)

    cohort = EX[_safe_id(cohort_id)]
    g.add((cohort, RDF.type, EX.FleetCohort))

    for k in ["dataset_type", "cohort_size", "window_days", "notes"]:
        if k in fleet and fleet.get(k) is not None:
            g.add((cohort, EX[k], Literal(str(fleet.get(k)))))

    # Fleet aggregate frames
    for a in (fleet.get("aggregate_attributes") or []):
        if not isinstance(a, dict):
            continue
        aid = a.get("id") or f"FLEET_AF_{a.get('attribute','UNK')}"
        refs.add(str(aid))
        n = EX[_safe_id(aid)]
        g.add((n, RDF.type, EX.FleetAttributeFrame))
        for kk, vv in a.items():
            if kk == "id" or vv is None:
                continue
            g.add((n, EX[kk], Literal(vv)))
        g.add((cohort, EX.hasFleetAttributeFrame, n))

    # Workload distribution (Alibaba apps)
    wd = fleet.get("workload_distribution")
    if isinstance(wd, dict) and wd:
        wid = wd.get("id") or "WL_DIST"
        refs.add(str(wid))
        n = EX[_safe_id(wid)]
        g.add((n, RDF.type, EX.WorkloadDistribution))
        for kk, vv in wd.items():
            if kk == "id" or vv is None:
                continue
            g.add((n, EX[kk], Literal(str(vv))))
        g.add((cohort, EX.hasWorkloadDistribution, n))

    # Drives + top signals only (compact)
    for d in (fleet.get("drives") or []):
        if not isinstance(d, dict):
            continue
        did_raw = d.get("drive_id") or d.get("disk_id") or d.get("id")
        if did_raw is None:
            continue
        did = _safe_id(did_raw)
        drv = EX[f"DRV_{did}"]
        g.add((drv, RDF.type, EX.Drive))
        g.add((drv, EX.drive_id, Literal(str(did_raw))))
        for k in ["ds", "app", "risk_score"]:
            if k in d and d.get(k) is not None and str(d.get(k)).strip() != "":
                g.add((drv, EX[k], Literal(d.get(k))))
        g.add((cohort, EX.hasDrive, drv))

        for s in (d.get("top_signals") or []):
            if not isinstance(s, dict):
                continue
            sid = s.get("id") or f"DRV_{did}_AF_{s.get('attribute','UNK')}"
            refs.add(str(sid))
            af = EX[_safe_id(sid)]
            g.add((af, RDF.type, EX.AttributeFrame))
            for kk in ["attribute", "median", "p95", "min", "max", "slope", "changepoint_idx", "outliers", "n"]:
                vv = s.get(kk)
                if vv is None:
                    continue
                g.add((af, EX[kk], Literal(vv)))
            g.add((drv, EX.hasAttributeFrame, af))

    ttl = g.serialize(format="turtle")
    return FleetKGArtifact(ttl=ttl, refs=set(r for r in refs if r))
