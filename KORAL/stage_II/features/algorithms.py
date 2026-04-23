#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flash controller algorithm/policy parsing."""

from __future__ import annotations
from typing import Any, Dict, List

def build_algorithms_ir(row: Dict[str, Any]) -> Dict[str, Any]:
    al = row.get("algorithms") or row.get("policies") or row.get("controller_policies")
    if al is None:
        return {}
    if isinstance(al, list):
        policies = [str(x).strip() for x in al if str(x).strip()]
    else:
        s = str(al).strip()
        if not s:
            return {}
        # allow semicolon separated
        policies = [p.strip() for p in s.split(";") if p.strip()]
    if not policies:
        return {}
    return {"algorithms": {"id": "AL_1", "policies": policies}}
