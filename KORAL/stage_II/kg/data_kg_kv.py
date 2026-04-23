#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data KG materialization for KV-SSD telemetry samples."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional, Set

try:
    import rdflib
    from rdflib import Graph, Namespace, Literal, RDF, URIRef
except Exception:
    rdflib = None
    Graph = None


@dataclass
class DataKGArtifact:
    ttl: Optional[str]
    refs: Set[str]


def _add_literal(g, subj, pred, val):
    if val is None or (isinstance(val, str) and not val.strip()):
        return
    g.add((subj, pred, Literal(val)))


def build_kv_data_kg(sample_id: str, ir: Dict[str, Any]) -> DataKGArtifact:
    """Materialize a TTL graph for one KV-SSD sample.

    Emits:
      :<sample_id>  a  koral:KVSample  ;
          koral:hasDevice     :<sample_id>/device  ;
          koral:hasFrame      :<sample_id>/frame/<name>  ... ;
          koral:observed      <scalar literals> .
    Each frame gets a node with median/p95/slope/etc., matching the Stage 1 KG
    linking class URIs under http://example.org/ssd/taxonomy/SSD/KVSSD/...
    """
    refs: Set[str] = {"KV_SAMPLE", "KV_DEVICE"}
    kv = ir.get("kv", {}) if isinstance(ir, dict) else {}

    if rdflib is None:
        for f in (kv.get("frames") or []):
            if f.get("id"):
                refs.add(f["id"])
        return DataKGArtifact(ttl=None, refs=refs)

    KORAL = Namespace("http://example.org/koral-data#")
    TAX = Namespace("http://example.org/ssd/taxonomy/SSD/KVSSD/")
    g = Graph()
    g.bind("koral", KORAL)
    g.bind("kvtax", TAX)

    s = KORAL[f"sample/{sample_id}"]
    g.add((s, RDF.type, KORAL.KVSample))
    _add_literal(g, s, KORAL.sampleId, str(sample_id))

    # Device
    dev = kv.get("device") or {}
    if dev:
        d = KORAL[f"sample/{sample_id}/device"]
        g.add((d, RDF.type, KORAL.KVDevice))
        for k, v in dev.items():
            if k == "id":
                continue
            _add_literal(g, d, KORAL[k], v)
        g.add((s, KORAL.hasDevice, d))

    # Scalars → observed properties (useful for SPARQL queries over KV_DEVICE)
    for k, v in (kv.get("scalar") or {}).items():
        if v is None:
            continue
        _add_literal(g, s, KORAL[k], v)

    # Value-size distribution
    vsd = kv.get("value_size_distribution")
    if isinstance(vsd, dict):
        vn = KORAL[f"sample/{sample_id}/vsize"]
        g.add((vn, RDF.type, KORAL.ValueSizeDistribution))
        g.add((vn, KORAL.type, Literal(vsd.get("type", "unknown"))))
        for k, v in vsd.items():
            if k == "type":
                continue
            if isinstance(v, dict):
                for i, (bk, bv) in enumerate(v.items()):
                    b = KORAL[f"sample/{sample_id}/vsize/bucket{i}"]
                    g.add((b, RDF.type, KORAL.ValueSizeBucket))
                    g.add((b, KORAL.label, Literal(str(bk))))
                    g.add((b, KORAL.fraction, Literal(bv)))
                    g.add((vn, KORAL.hasBucket, b))
            else:
                _add_literal(g, vn, KORAL[k], v)
        g.add((s, KORAL.hasValueSizeDistribution, vn))

    # Time-series attribute frames
    _SIGNAL_TO_TAX = {
        "inline_ratio":              TAX["Behavioral/InlineRatio"],
        "inline_to_regular_rate":    TAX["Behavioral/InlineToRegularTransition"],
        "mapping_entry_churn":       TAX["Behavioral/MappingEntryChurn"],
        "translation_page_pressure": TAX["Behavioral/TranslationPagePressure"],
        "cmt_hit_rate":              TAX["Behavioral/CMTHitRateDynamics"],
        "cmt_eviction_rate":         TAX["Behavioral/CMTHitRateDynamics"],
        "cmt_size_entries":          TAX["KVFTL/CachedMappingTable"],
        "tail_latency_p999_us":      TAX["Performance/TailLatency"],
        "read_latency_p99_us":       TAX["Performance/TailLatency"],
        "read_latency_p50_us":       TAX["Performance/TailLatency"],
        "throughput_kops":           TAX["Performance/Throughput"],
        "write_amplification":       TAX["Performance/WriteAmplification"],
        "read_amplification":        TAX["Performance/ReadAmplification"],
        "value_size_bytes":          TAX["Workload/ValueSize"],
        "key_size_bytes":            TAX["Workload/KeySize"],
        "read_write_ratio":          TAX["Workload/ReadWriteRatio"],
        "access_skew":               TAX["Workload/AccessSkew"],
    }

    for f in (kv.get("frames") or []):
        fid = f.get("id")
        if not fid:
            continue
        refs.add(fid)
        fn = KORAL[f"sample/{sample_id}/frame/{f['attribute']}"]
        g.add((fn, RDF.type, KORAL.AttributeFrame))
        g.add((fn, KORAL.attribute, Literal(f["attribute"])))
        if f.get("unit"):
            g.add((fn, KORAL.unit, Literal(f["unit"])))
        tax_class = _SIGNAL_TO_TAX.get(f["attribute"])
        if tax_class is not None:
            g.add((fn, KORAL.maps_to, tax_class))
        for k in ["median", "p95", "min", "max", "slope", "changepoint_idx", "outliers", "n", "coverage"]:
            v = f.get(k)
            if v is None:
                continue
            g.add((fn, KORAL[k], Literal(v)))
        g.add((s, KORAL.hasFrame, fn))

    ttl = g.serialize(format="turtle")
    return DataKGArtifact(ttl=ttl, refs=refs)
