#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RocksDB LOG / db_bench / trace_replay stats adapter.

Pulls the lines RocksDB writes into its LOG file and a trace_replay summary,
and converts them into the KV telemetry row schema.
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Any, Dict


_PAT_BYTES_WRITTEN  = re.compile(r"Cumulative writes:.*?(\d[\d.]*)\s*GB written", re.IGNORECASE)
_PAT_BYTES_READ     = re.compile(r"Cumulative reads:.*?(\d[\d.]*)\s*GB read", re.IGNORECASE)
_PAT_WA             = re.compile(r"write[- ]amplification[:\s]+(\d+\.?\d*)", re.IGNORECASE)
_PAT_RA             = re.compile(r"read[- ]amplification[:\s]+(\d+\.?\d*)", re.IGNORECASE)
_PAT_COMPACT_S      = re.compile(r"Compact\(s\)\s*:\s*(\d+\.?\d*)", re.IGNORECASE)
_PAT_P50            = re.compile(r"Percentile.*?50th[^0-9]*(\d+\.?\d*)\s*us", re.IGNORECASE | re.DOTALL)
_PAT_P99            = re.compile(r"Percentile.*?99th[^0-9]*(\d+\.?\d*)\s*us", re.IGNORECASE | re.DOTALL)
_PAT_P999           = re.compile(r"Percentile.*?99\.9[^0-9]*(\d+\.?\d*)\s*us", re.IGNORECASE | re.DOTALL)
_PAT_OPS_SEC        = re.compile(r"([\d.]+)\s*ops/sec", re.IGNORECASE)


def _search_float(pat: re.Pattern, text: str):
    m = pat.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def parse_rocksdb_log(path: str | Path) -> Dict[str, Any]:
    """Parse a RocksDB LOG or db_bench output file into a KV telemetry row."""
    p = Path(path)
    text = p.read_text(encoding="utf-8", errors="ignore")
    row: Dict[str, Any] = {"source": "rocksdb", "source_path": str(p), "kv_ftl": "rocksdb-on-block"}

    ops = _search_float(_PAT_OPS_SEC, text)
    if ops is not None:
        row["throughput_kops"] = ops / 1000.0

    p50 = _search_float(_PAT_P50, text)
    if p50 is not None:
        row["read_latency_p50_us"] = p50
    p99 = _search_float(_PAT_P99, text)
    if p99 is not None:
        row["read_latency_p99_us"] = p99
    p999 = _search_float(_PAT_P999, text)
    if p999 is not None:
        row["tail_latency_p999_us"] = p999

    wa = _search_float(_PAT_WA, text)
    if wa is not None:
        row["write_amplification"] = wa
    ra = _search_float(_PAT_RA, text)
    if ra is not None:
        row["read_amplification"] = ra

    # If cumulative bytes are present, derive an approximate WA when db_bench didn't compute it.
    if "write_amplification" not in row:
        gb_w = _search_float(_PAT_BYTES_WRITTEN, text)
        gb_user = _search_float(re.compile(r"User writes:.*?(\d+\.?\d*)\s*GB", re.IGNORECASE), text)
        if gb_w and gb_user and gb_user > 0:
            row["write_amplification"] = gb_w / gb_user

    return row
