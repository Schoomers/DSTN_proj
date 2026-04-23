#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KVPack-style simulator output adapter.

Expects a JSON or CSV file where each record carries KV-FTL internals
directly (inline ratio, CMT hit rate, mapping churn, etc.). This is the
fast path — no heuristics, since the simulator emits the signals we need.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List


_ALLOWED_KEYS = {
    "inline_ratio", "inline_to_regular_rate", "mapping_entry_churn",
    "translation_page_pressure", "cmt_hit_rate", "cmt_eviction_rate",
    "cmt_size_entries", "value_size_bytes", "key_size_bytes",
    "read_write_ratio", "update_intensity", "access_skew",
    "throughput_kops", "read_latency_p50_us", "read_latency_p99_us",
    "tail_latency_p999_us", "write_amplification", "read_amplification",
    "gc_overhead_pct", "value_size_dist",
    "vendor", "model", "firmware", "capacity_gb", "interface", "kv_ftl",
    "sample_id", "window_id",
}


def _clean(row: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in row.items() if k in _ALLOWED_KEYS and v is not None}
    out.setdefault("source", "kvpack")
    out.setdefault("kv_ftl", "KVPack")
    return out


def parse_kvpack_json(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = [data]
    return [_clean(r) for r in data if isinstance(r, dict)]


def parse_kvpack_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(_clean(obj))
    return rows
