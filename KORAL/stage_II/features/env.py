#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Environmental feature parsing."""

from __future__ import annotations
from typing import Any, Dict, Optional

def build_env_ir(row: Dict[str, Any]) -> Dict[str, Any]:
    """Accepts env columns if present and returns IR-friendly structure."""
    # We keep this permissive because env_effects.csv can vary by paper.
    keys = [
        "temperature_c", "relative_humidity_pct",
        "vibration_freq_hz", "vibration_amp_g",
        "throughput_change_pct", "tail_latency_change_pct",
        "study", "condition_id",
    ]
    env = {}
    for k in keys:
        if k in row and row.get(k) is not None and str(row.get(k)).strip() != "":
            env[k] = row.get(k)
    if not env:
        return {}
    env["id"] = row.get("env_id") or row.get("condition_id") or "ENV_1"
    return {"env": env}
