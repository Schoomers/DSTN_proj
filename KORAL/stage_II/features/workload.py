#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workload feature parsing."""

from __future__ import annotations
import re
from typing import Any, Dict

def parse_app_tag(app: Any) -> Dict[str, Any]:
    if app is None:
        return {}
    s = str(app).strip()
    if not s:
        return {}
    return {"id": "WL_APP", "type": "app_tag", "value": s}

def parse_fio_job(text: Any) -> Dict[str, Any]:
    if text is None:
        return {}
    s = str(text).strip()
    if not s:
        return {}
    # Very light parsing: capture rw, rwmixread, bs, iodepth, numjobs
    out = {"id": "WL_FIO", "type": "fio"}
    for key in ["rw", "rwmixread", "bs", "iodepth", "numjobs", "iodepth_batch", "rate_iops", "runtime"]:
        m = re.search(rf"^{re.escape(key)}\s*=\s*([^\n\r]+)", s, flags=re.MULTILINE)
        if m:
            out[key] = m.group(1).strip()
    out["raw"] = s
    return out

def build_workload_ir(row: Dict[str, Any]) -> Dict[str, Any]:
    # Alibaba "app" tag
    if "app" in row:
        w = parse_app_tag(row.get("app"))
        if w:
            return {"workload": w}
    # FIO jobfile text
    if "fio_job" in row:
        w = parse_fio_job(row.get("fio_job"))
        if w:
            return {"workload": w}
    if "workload" in row:
        # generic
        s = str(row.get("workload")).strip()
        if s:
            return {"workload": {"id": "WL_GENERIC", "type": "text", "value": s}}
    return {}
