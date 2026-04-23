#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KV-SSD feature extraction into an intermediate representation (IR).

Captures the behavioral/workload/performance/system signals required by the
KV-SSD diagnosis ontology:
  - workload: key size, value size distribution, R/W ratio, update intensity, access skew
  - cmt:      hit rate, size (entries), eviction rate
  - mapping:  inline ratio, inline-to-regular transition rate, mapping entry churn,
              translation page pressure
  - perf:     throughput (KOPS), read/write latency (p50/p99/p99.9), WA/RA

A sample row may come from any telemetry source — see stage_II/adapters/ for
MQSim, RocksDB, and KVPack converters that produce rows in this schema.
"""

from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from stage_II.features.smart import (
    parse_series,
    robust_stats,
    trend_slope,
    changepoint_heuristic,
    outlier_count,
)


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None


def _pick(row: Dict[str, Any], *keys: str) -> Optional[Any]:
    for k in keys:
        if k in row and row[k] is not None and str(row[k]).strip() != "":
            return row[k]
    return None


@dataclass
class KVFrame:
    """Time-series attribute frame (parallels SMART's SmartFrame)."""
    name: str
    series: List[float]
    stats: Dict[str, Any]
    slope: Optional[float]
    changepoint_idx: Optional[int]
    outliers: int
    unit: Optional[str] = None

    def to_ir(self) -> Dict[str, Any]:
        return {
            "id": f"KV_{self.name}",
            "attribute": self.name,
            "unit": self.unit,
            "n": self.stats.get("n", 0),
            "median": self.stats.get("median"),
            "p95": self.stats.get("p95"),
            "min": self.stats.get("min"),
            "max": self.stats.get("max"),
            "slope": self.slope,
            "changepoint_idx": self.changepoint_idx,
            "outliers": self.outliers,
            "coverage": self.stats.get("coverage", 0.0),
        }


# Each (attribute_name, list_of_row_keys, unit) maps a KV signal to possible
# column aliases in the input. First non-empty alias wins.
_SIGNAL_MAP = [
    # System / mapping / CMT (behavioral)
    ("inline_ratio",                 ["inline_ratio", "kv_inline_ratio"],                        "ratio"),
    ("inline_to_regular_rate",       ["inline_to_regular_rate", "inline2regular", "itr_rate"],   "events/sec"),
    ("mapping_entry_churn",          ["mapping_entry_churn", "mapping_churn"],                   "entries/sec"),
    ("translation_page_pressure",    ["translation_page_pressure", "tp_pressure"],               "ratio"),
    ("cmt_hit_rate",                 ["cmt_hit_rate", "cached_mapping_hit_rate"],                "ratio"),
    ("cmt_eviction_rate",            ["cmt_eviction_rate", "cmt_evict_rate"],                    "entries/sec"),
    ("cmt_size_entries",             ["cmt_size_entries", "cmt_size"],                           "entries"),
    # Workload
    ("value_size_bytes",             ["value_size_bytes", "value_size", "vsize"],                "bytes"),
    ("key_size_bytes",               ["key_size_bytes", "key_size", "ksize"],                    "bytes"),
    ("read_write_ratio",             ["read_write_ratio", "rw_ratio"],                           "ratio"),
    ("update_intensity",             ["update_intensity", "update_rate"],                        "ratio"),
    ("access_skew",                  ["access_skew", "zipfian_s", "skew"],                       "zipf_s"),
    # Performance
    ("throughput_kops",              ["throughput_kops", "throughput"],                          "KOPS"),
    ("read_latency_p50_us",          ["read_latency_p50_us", "read_p50"],                        "us"),
    ("read_latency_p99_us",          ["read_latency_p99_us", "read_p99"],                        "us"),
    ("tail_latency_p999_us",         ["tail_latency_p999_us", "p999_latency_us", "tail_p999"],   "us"),
    ("write_amplification",          ["write_amplification", "wa"],                              "x"),
    ("read_amplification",           ["read_amplification", "ra"],                               "x"),
    ("gc_overhead_pct",              ["gc_overhead_pct", "gc_pct"],                              "pct"),
]


def _build_frame(name: str, raw_val: Any, unit: Optional[str]) -> Optional[KVFrame]:
    series = parse_series(raw_val)
    if not series:
        return None
    stats = robust_stats(series)
    return KVFrame(
        name=name,
        series=series,
        stats=stats,
        slope=trend_slope(series),
        changepoint_idx=changepoint_heuristic(series),
        outliers=outlier_count(series),
        unit=unit,
    )


def _value_size_distribution(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Parse a value-size distribution, either as explicit buckets JSON or a series."""
    raw = _pick(row, "value_size_dist", "value_size_histogram")
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    # Try JSON object of bucket -> fraction
    if s.startswith("{"):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return {"type": "buckets", "buckets": obj}
        except Exception:
            pass
    # Fall back to series statistics
    series = parse_series(raw)
    if series:
        st = robust_stats(series)
        return {
            "type": "series_stats",
            "median": st.get("median"),
            "p95": st.get("p95"),
            "min": st.get("min"),
            "max": st.get("max"),
            "n": st.get("n"),
        }
    return None


def build_kv_ir(row: Dict[str, Any]) -> Dict[str, Any]:
    """Build the KV-SSD IR for a single telemetry window/sample."""
    frames: List[KVFrame] = []
    scalar: Dict[str, Any] = {}

    for name, aliases, unit in _SIGNAL_MAP:
        raw = _pick(row, *aliases)
        if raw is None:
            continue
        # If multi-value (series), build an attribute frame; else record scalar.
        series = parse_series(raw)
        if len(series) >= 2:
            f = _build_frame(name, raw, unit)
            if f:
                frames.append(f)
                # Also export last-point scalar for quick prompt consumption.
                scalar[name] = series[-1]
        else:
            fv = _to_float(raw)
            scalar[name] = fv if fv is not None else str(raw)

    device = {
        "id": "KV_DEVICE",
        "vendor": _pick(row, "vendor"),
        "model": _pick(row, "model", "device_model"),
        "firmware": _pick(row, "firmware", "firmware_version"),
        "capacity_gb": _to_float(_pick(row, "capacity_gb")),
        "interface": _pick(row, "interface", "interface_type"),
        "kv_ftl": _pick(row, "kv_ftl", "ftl_variant") or "KVPack-style",
    }
    device = {k: v for k, v in device.items() if v is not None}

    vsize_dist = _value_size_distribution(row)

    ir: Dict[str, Any] = {
        "kv": {
            "id": "KV_SAMPLE",
            "device": device,
            "scalar": scalar,
            "value_size_distribution": vsize_dist,
            "frames": [f.to_ir() for f in frames],
        }
    }
    return ir


def collect_kv_refs(ir: Dict[str, Any]) -> List[str]:
    """Return reference IDs that prompts may cite."""
    refs: List[str] = ["KV_SAMPLE", "KV_DEVICE"]
    kv = ir.get("kv", {}) if isinstance(ir, dict) else {}
    for f in kv.get("frames", []) or []:
        rid = f.get("id")
        if rid:
            refs.append(rid)
    return refs


def kv_query_terms(ir: Dict[str, Any]) -> List[str]:
    """Select KG-retrieval terms based on which KV signals are present and anomalous."""
    terms = ["KV-SSD", "KV-FTL", "LSM", "compaction"]
    kv = ir.get("kv", {}) if isinstance(ir, dict) else {}
    scalar = kv.get("scalar", {}) or {}
    frames_by_name = {f["attribute"]: f for f in (kv.get("frames") or [])}

    def _present(*names: str) -> bool:
        return any(n in scalar or n in frames_by_name for n in names)

    if _present("inline_ratio", "inline_to_regular_rate"):
        terms += ["inline", "Inline Ratio", "Inline-to-Regular Transition"]
    if _present("cmt_hit_rate", "cmt_eviction_rate", "cmt_size_entries"):
        terms += ["CMT", "Cached Mapping Table", "CMT Hit Rate"]
    if _present("mapping_entry_churn"):
        terms += ["Mapping Entry Churn", "mapping entry"]
    if _present("translation_page_pressure"):
        terms += ["Translation Page", "Translation Page Pressure"]
    if _present("value_size_bytes") or kv.get("value_size_distribution"):
        terms += ["Value Size", "KV Separation"]
    if _present("tail_latency_p999_us", "read_latency_p99_us"):
        terms += ["Tail Latency", "99th Percentile"]
    if _present("write_amplification"):
        terms += ["Write Amplification", "compaction", "Garbage Collection"]
    if _present("read_amplification"):
        terms += ["Read Amplification"]
    if _present("access_skew"):
        terms += ["Access Skew", "Zipfian"]
    return terms


# Lightweight rule-based flags used by prompts/diagnosis as hypotheses.
# (These are NOT claims — they just prime the LLM with observations.)
def kv_observations(ir: Dict[str, Any]) -> List[Dict[str, Any]]:
    kv = ir.get("kv", {}) if isinstance(ir, dict) else {}
    scalar = kv.get("scalar", {}) or {}
    frames = {f["attribute"]: f for f in (kv.get("frames") or [])}
    obs: List[Dict[str, Any]] = []

    def _add(kind: str, signal: str, detail: str):
        ref = f"KV_{signal}" if signal in frames else "KV_SAMPLE"
        obs.append({"kind": kind, "signal": signal, "detail": detail, "ref": ref})

    hit = scalar.get("cmt_hit_rate")
    if isinstance(hit, (int, float)) and hit < 0.85:
        _add("cmt_pressure", "cmt_hit_rate", f"CMT hit rate is low ({hit:.2f}); translation-page reads likely increasing.")

    pressure = scalar.get("translation_page_pressure")
    if isinstance(pressure, (int, float)) and pressure > 0.5:
        _add("tp_pressure", "translation_page_pressure", f"Translation-page pressure is elevated ({pressure:.2f}).")

    inline = scalar.get("inline_ratio")
    if isinstance(inline, (int, float)) and inline < 0.4:
        _add("inline_collapse", "inline_ratio", f"Inline ratio is low ({inline:.2f}); most entries are regular, raising CMT working-set size.")

    itr = scalar.get("inline_to_regular_rate")
    if isinstance(itr, (int, float)) and itr > 0:
        _add("inline_to_regular", "inline_to_regular_rate", f"Inline→regular transitions observed ({itr}/s); mapping-entry churn will follow.")

    churn = scalar.get("mapping_entry_churn")
    if isinstance(churn, (int, float)) and churn > 0:
        f = frames.get("mapping_entry_churn")
        slope = f.get("slope") if f else None
        if slope is not None and slope > 0:
            _add("churn_rising", "mapping_entry_churn", f"Mapping-entry churn is rising (slope={slope:.3g}).")

    tp999 = scalar.get("tail_latency_p999_us")
    if isinstance(tp999, (int, float)):
        f = frames.get("tail_latency_p999_us")
        slope = f.get("slope") if f else None
        if slope is not None and slope > 0:
            _add("tail_rising", "tail_latency_p999_us", f"p99.9 read latency trending up (slope={slope:.3g} us/step).")

    return obs
