#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data KG materialization (lightweight)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

try:
    import rdflib
    from rdflib import Graph, Namespace, Literal, RDF, URIRef
except Exception:  # pragma: no cover
    rdflib = None
    Graph = None

@dataclass
class DataKGArtifact:
    ttl: Optional[str]
    refs: Set[str]

def build_data_kg(sample_id: str, ir: Dict[str, Any]) -> DataKGArtifact:
    """Create a tiny TTL graph for a sample. Returns TTL string + reference IDs."""
    refs: Set[str] = set()
    if rdflib is None:
        # no rdflib; just keep refs from IR
        for sec in ["smart", "env", "workload", "flash_type", "algorithms"]:
            if sec in ir:
                if sec == "smart":
                    for af in ir["smart"]:
                        refs.add(af.get("id", ""))
                else:
                    obj = ir.get(sec)
                    if isinstance(obj, dict) and "id" in obj:
                        refs.add(str(obj["id"]))
        return DataKGArtifact(ttl=None, refs=set(r for r in refs if r))

    EX = Namespace("http://example.org/koral-data#")
    g = Graph()
    g.bind("ex", EX)

    s = EX[str(sample_id)]
    g.add((s, RDF.type, EX.Sample))

    # SMART frames
    for af in ir.get("smart", []):
        af_id = af.get("id")
        if not af_id:
            continue
        refs.add(af_id)
        n = EX[af_id]
        g.add((n, RDF.type, EX.AttributeFrame))
        g.add((n, EX.attribute, Literal(af.get("attribute"))))
        g.add((n, EX.coverage, Literal(float(af.get("coverage", 0.0)))))
        for k in ["median", "p95", "min", "max", "slope", "changepoint_idx", "outliers", "n"]:
            v = af.get(k, None)
            if v is None:
                continue
            g.add((n, EX[k], Literal(v)))
        g.add((s, EX.hasAttributeFrame, n))

    # Optional sections
    for sec in ["env", "workload", "flash_type", "algorithms"]:
        obj = ir.get(sec)
        if not obj:
            continue
        obj_id = obj.get("id", sec.upper())
        refs.add(obj_id)
        n = EX[obj_id]
        g.add((n, RDF.type, EX[sec.title().replace("_","")]))
        for k, v in obj.items():
            if k == "id":
                continue
            g.add((n, EX[k], Literal(str(v))))
        g.add((s, EX[f"has_{sec}"], n))

    ttl = g.serialize(format="turtle")
    return DataKGArtifact(ttl=ttl, refs=set(r for r in refs if r))
