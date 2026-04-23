#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SMART feature extraction into an intermediate representation (IR) + Data-KG-friendly dicts."""

from __future__ import annotations
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_LISTISH_RE = re.compile(r"^\s*\[.*\]\s*$")

def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return None
        return float(x)
    except Exception:
        return None

def parse_series(val: Any) -> List[float]:
    """Parse a 30-day series stored as JSON list, or delimited string, or scalar."""
    if val is None:
        return []
    if isinstance(val, (list, tuple, np.ndarray)):
        out = []
        for v in val:
            fv = _to_float(v)
            if fv is not None:
                out.append(fv)
        return out

    s = str(val).strip()
    if not s:
        return []
    # JSON list
    if _LISTISH_RE.match(s):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return [float(v) for v in arr if _to_float(v) is not None]
        except Exception:
            pass
    # delimited list
    if ";" in s:
        parts = s.split(";")
    elif "," in s and any(ch.isdigit() for ch in s):
        parts = s.split(",")
    else:
        parts = [s]
    out = []
    for p in parts:
        fv = _to_float(p)
        if fv is not None:
            out.append(fv)
    return out

def robust_stats(x: List[float]) -> Dict[str, Any]:
    """Compute robust summary stats for a numeric series."""
    if not x:
        return {"n": 0, "coverage": 0.0}
    arr = np.asarray(x, dtype=float)
    n = int(arr.size)
    med = float(np.median(arr))
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    iqr = q75 - q25
    p95 = float(np.percentile(arr, 95))
    p05 = float(np.percentile(arr, 5))
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    # MAD (median absolute deviation)
    mad = float(np.median(np.abs(arr - med))) if n > 0 else 0.0
    return {
        "n": n,
        "coverage": 1.0,  # series parser removes missing
        "median": med,
        "p95": p95,
        "p05": p05,
        "min": mn,
        "max": mx,
        "q25": q25,
        "q75": q75,
        "iqr": float(iqr),
        "mad": mad,
    }

def trend_slope(x: List[float]) -> Optional[float]:
    """Simple linear slope over index (units: value/day if daily)."""
    if x is None or len(x) < 3:
        return None
    y = np.asarray(x, dtype=float)
    t = np.arange(len(y), dtype=float)
    try:
        slope = float(np.polyfit(t, y, 1)[0])
        return slope
    except Exception:
        return None

def changepoint_heuristic(x: List[float]) -> Optional[int]:
    """Return an approximate changepoint index via max absolute first-difference."""
    if x is None or len(x) < 4:
        return None
    y = np.asarray(x, dtype=float)
    diffs = np.abs(np.diff(y))
    if diffs.size == 0:
        return None
    idx = int(np.argmax(diffs) + 1)  # changepoint between idx-1 and idx
    return idx

def outlier_count(x: List[float], z: float = 3.5) -> int:
    """Count robust outliers using median and MAD."""
    if x is None or len(x) < 4:
        return 0
    y = np.asarray(x, dtype=float)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    if mad == 0:
        return 0
    robust_z = 0.6745 * (y - med) / mad
    return int(np.sum(np.abs(robust_z) > z))

@dataclass
class SmartFrame:
    name: str
    series: List[float]
    stats: Dict[str, Any]
    slope: Optional[float]
    changepoint_idx: Optional[int]
    outliers: int

    def to_ir(self) -> Dict[str, Any]:
        return {
            "id": f"AF_{self.name}",
            "attribute": self.name,
            "n": self.stats.get("n", 0),
            "median": self.stats.get("median", None),
            "p95": self.stats.get("p95", None),
            "min": self.stats.get("min", None),
            "max": self.stats.get("max", None),
            "slope": self.slope,
            "changepoint_idx": self.changepoint_idx,
            "outliers": self.outliers,
            "coverage": self.stats.get("coverage", 0.0),
        }

def build_smart_ir(row: Dict[str, Any], smart_cols: List[str]) -> Dict[str, Any]:
    """Create an IR dict for SMART attributes from an input sample row."""
    frames: List[SmartFrame] = []
    for c in smart_cols:
        if c not in row:
            continue
        s = parse_series(row.get(c))
        st = robust_stats(s)
        frame = SmartFrame(
            name=c,
            series=s,
            stats=st,
            slope=trend_slope(s),
            changepoint_idx=changepoint_heuristic(s),
            outliers=outlier_count(s),
        )
        frames.append(frame)

    return {
        "smart": [f.to_ir() for f in frames]
    }

def infer_smart_columns(row_keys: List[str]) -> List[str]:
    """Infer SMART columns like r_5, r_9, ... from a CSV header."""
    cols = []
    for k in row_keys:
        if re.fullmatch(r"r_\d+", str(k)):
            cols.append(str(k))
    return sorted(cols, key=lambda x: int(x.split("_")[1]))
