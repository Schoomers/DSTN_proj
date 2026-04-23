#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Robust JSON parsing for LLM outputs."""

from __future__ import annotations
import json
import re
from typing import Any, Dict, Optional

def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """Try to extract the first JSON object from a model response."""
    if text is None:
        return None
    s = text.strip()
    # fast path
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # heuristic: find outermost {...}
    m = re.search(r"\{[\s\S]*\}", s)
    if not m:
        return None
    candidate = m.group(0)
    # remove trailing code fences if present
    candidate = candidate.strip().strip("`")
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None
    return None
