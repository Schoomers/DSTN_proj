#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Twitter cache-trace adapter (OSDI '20 Yang et al., cluster00X files).

Reads a single cluster's trace (decompressed CSV with columns:
timestamp, key, key_size, value_size, client_id, operation, TTL), time-windows
it, and emits one KV telemetry row per window in the schema consumed by
stage_II.features.kv.

Signals produced per window:
  REAL (from the trace itself):
    - key_size_bytes (series of per-window mean)
    - value_size_bytes (series of per-window mean)
    - value_size_dist (bucket histogram across all windows)
    - read_write_ratio (gets / total ops per window)
    - update_intensity (writes-on-existing-key / total writes)
    - access_skew (top-1% key share as a skew proxy per window)
    - throughput_kops (reqs/sec / 1000)

  ESTIMATED under a KVPack-style KV-SSD assumption (marked via meta):
    - inline_ratio: fraction of ops whose (key+value+8B meta) <= frame threshold
    - mapping_entry_churn: set rate per second

NOT PRODUCED (require a KV-SSD internal simulator like MoKE+KVPack):
    - cmt_hit_rate, cmt_eviction_rate, translation_page_pressure,
      inline_to_regular_rate, read/write amplification, tail latencies.

Callers that want those signals must supplement with MoKE/KVPack output.
"""

from __future__ import annotations
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **_):
        return it


# Default KVPack frame threshold. KVPack paper Table II / §IV-B:
# a KV pair is inline if it fits in one frame (≈32B regular mapping entry size),
# extendable to multiple frames under KVPack-D. Here we use 64B as a permissive
# single/double-frame inline threshold for the estimate.
DEFAULT_FRAME_THRESHOLD_BYTES = 64
MAPPING_METADATA_OVERHEAD_BYTES = 8


def _bucketize_value_size(v: int) -> str:
    if v <= 64:  return "<=64B"
    if v <= 256: return "<=256B"
    if v <= 1024: return "<=1KB"
    if v <= 4096: return "<=4KB"
    return ">4KB"


def parse_twitter_trace(
    path: str | Path,
    n_windows: int = 8,
    max_rows: int = 1_000_000,
    frame_threshold_bytes: int = DEFAULT_FRAME_THRESHOLD_BYTES,
) -> Dict[str, Any]:
    """Parse a Twitter cache-trace file into a single KV telemetry row with
    per-window series. Returns a dict in stage_II.features.kv input schema."""
    p = Path(path)
    cluster_name = p.name  # e.g. "cluster018"

    # First pass: collect all rows (capped) and the timestamp range so we
    # can bucket into n_windows equal-time windows.
    rows: List[Tuple[int, str, int, int, str]] = []
    t_min = None
    t_max = None

    with p.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        read_bar = tqdm(total=max_rows, desc=f"read {cluster_name}", unit="row",
                        dynamic_ncols=True, mininterval=0.5)
        for i, r in enumerate(reader):
            if i >= max_rows:
                break
            if len(r) < 7:
                continue
            try:
                ts = int(r[0])
                key = r[1]
                ks = int(r[2])
                vs = int(r[3])
                op = r[5]
            except ValueError:
                continue
            rows.append((ts, key, ks, vs, op))
            if t_min is None or ts < t_min:
                t_min = ts
            if t_max is None or ts > t_max:
                t_max = ts
            read_bar.update(1)
        read_bar.close()

    if not rows:
        raise ValueError(f"No parseable rows in {p}")

    if t_max <= t_min:
        # Fallback: split by index count instead of timestamp
        window_of = lambda ts, idx: min(n_windows - 1, (idx * n_windows) // len(rows))
    else:
        span = t_max - t_min
        window_size_s = max(1.0, span / n_windows)
        def window_of(ts: int, idx: int) -> int:
            return min(n_windows - 1, int((ts - t_min) / window_size_s))

    # Per-window aggregates.
    wins = [dict(
        n=0, n_get=0, n_set=0, n_other=0,
        key_size_sum=0, val_size_sum=0, val_count=0,
        inline_fit=0, inline_total=0,
        key_counter=Counter(),
        seen_keys=set(),
        writes_on_existing=0,
    ) for _ in range(n_windows)]

    # Global value-size bucket histogram.
    vsize_bucket = Counter()
    global_seen_keys = set()

    for idx, (ts, key, ks, vs, op) in enumerate(rows):
        w = window_of(ts, idx)
        W = wins[w]
        W["n"] += 1
        W["key_size_sum"] += ks
        if vs > 0:
            W["val_size_sum"] += vs
            W["val_count"] += 1
            vsize_bucket[_bucketize_value_size(vs)] += 1
        op_l = op.lower()
        if op_l in ("get", "gets"):
            W["n_get"] += 1
        elif op_l in ("set", "add", "replace", "cas", "append", "prepend", "incr", "decr", "delete"):
            W["n_set"] += 1
            if key in global_seen_keys:
                W["writes_on_existing"] += 1
        else:
            W["n_other"] += 1
        # Inline estimate: (key + value + metadata) <= frame threshold.
        if (ks + vs + MAPPING_METADATA_OVERHEAD_BYTES) <= frame_threshold_bytes:
            W["inline_fit"] += 1
        W["inline_total"] += 1
        W["key_counter"][key] += 1
        global_seen_keys.add(key)

    # Derive series.
    def _series(extract):
        return [extract(W) if W["n"] > 0 else 0.0 for W in wins]

    key_size_series = _series(lambda W: W["key_size_sum"] / max(1, W["n"]))
    val_size_series = _series(lambda W: (W["val_size_sum"] / W["val_count"]) if W["val_count"] > 0 else 0.0)
    rw_ratio_series = _series(lambda W: W["n_get"] / max(1, W["n"]))
    update_intensity_series = _series(
        lambda W: W["writes_on_existing"] / max(1, W["n_set"]) if W["n_set"] > 0 else 0.0
    )
    inline_ratio_series = _series(lambda W: W["inline_fit"] / max(1, W["inline_total"]))

    # Access skew: share of top-1% of keys in that window's request volume.
    def _top_share(W, pct=0.01):
        if W["n"] == 0:
            return 0.0
        total = W["n"]
        k = max(1, int(len(W["key_counter"]) * pct))
        top = sum(c for _, c in W["key_counter"].most_common(k))
        return top / total
    access_skew_series = [_top_share(W) for W in wins]

    # Throughput estimate per window (kops), using window wall-clock span when
    # timestamps are usable, otherwise a dimensionless per-window ops/1000.
    if t_min is not None and t_max is not None and t_max > t_min:
        win_span = max(1.0, (t_max - t_min) / n_windows)
    else:
        win_span = 1.0
    throughput_series = [W["n"] / win_span / 1000.0 for W in wins]

    # Mapping-entry churn estimate (KVPack §IV-A sense: entries inserted/updated/sec).
    mapping_churn_series = [W["n_set"] / win_span for W in wins]

    # Overall bucket histogram, normalized to fractions.
    total_vcount = sum(vsize_bucket.values()) or 1
    value_size_dist = {k: round(v / total_vcount, 4) for k, v in vsize_bucket.items()}

    row: Dict[str, Any] = {
        "sample_id": f"twitter_{cluster_name}",
        "source": "twitter_trace",
        "kv_ftl": "twitter-trace-estimated-under-KVPack",
        "model": cluster_name,
        "vendor": "Twitter/Twemcache OSDI20",
        "interface": "memcache",
        "anomaly_kind": None,

        # Real workload-side signals:
        "key_size_bytes": json.dumps(key_size_series),
        "value_size_bytes": json.dumps(val_size_series),
        "value_size_dist": json.dumps(value_size_dist),
        "read_write_ratio": json.dumps(rw_ratio_series),
        "update_intensity": json.dumps(update_intensity_series),
        "access_skew": json.dumps(access_skew_series),
        "throughput_kops": json.dumps(throughput_series),

        # Estimated under KVPack-style assumptions:
        "inline_ratio": json.dumps(inline_ratio_series),
        "mapping_entry_churn": json.dumps(mapping_churn_series),

        # Meta: flag which signals are estimated vs real.
        "estimation_note": (
            f"inline_ratio and mapping_entry_churn are estimated under a "
            f"KVPack frame_threshold={frame_threshold_bytes}B. "
            f"CMT dynamics, translation-page pressure, inline-to-regular "
            f"transitions, and WA/RA/tail-latency are NOT available from "
            f"the trace alone and require MoKE+KVPack or a KV-SSD simulator."
        ),
        "trace_rows_processed": len(rows),
        "n_windows": n_windows,
        "time_span_s": (t_max - t_min) if (t_min is not None and t_max is not None) else None,
    }
    return row


def parse_twitter_cluster(cluster_path: str | Path, **kwargs) -> List[Dict[str, Any]]:
    """List wrapper used by the CLI adapter dispatch."""
    return [parse_twitter_trace(cluster_path, **kwargs)]
