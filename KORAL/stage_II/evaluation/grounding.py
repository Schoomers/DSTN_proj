#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Grounding metrics: FiP and CFV."""

from __future__ import annotations
import re
from typing import Any, Dict, List, Set, Tuple

_DIR_RE = re.compile(r"(increase|decrease)", re.IGNORECASE)

def faithfulness_precision(output_json: Dict[str, Any], available_refs: Set[str]) -> float:
    """FiP = supported_atomic_claims / atomic_claims."""
    claims = output_json.get("atomic_claims", [])
    if not isinstance(claims, list) or len(claims) == 0:
        return 0.0
    supported = 0
    total = 0
    for c in claims:
        if not isinstance(c, dict):
            continue
        total += 1
        sup = c.get("support", [])
        if not isinstance(sup, list) or len(sup) == 0:
            continue
        ok = True
        for ref in sup:
            if ref is None:
                ok = False
                break
            r = str(ref).strip()
            # accept prefixes like "IR:" or "LIT:"
            r = r.replace("IR:", "").replace("ENV:", "").replace("LIT:", "")
            if r not in available_refs and not r.startswith("LIT_"):
                ok = False
                break
        if ok:
            supported += 1
    return float(supported / total) if total else 0.0

def counterfactual_validity(output_json: Dict[str, Any], direction_lookup: Dict[str, Dict[str, str]] | None = None) -> float:
    """CFV = statements with evidence-consistent direction / statements.

    direction_lookup maps evidence_id -> {"effect_direction": "increase|decrease|unclear"}.
    If not provided, we fallback to a weak check: statement has evidence and a direction token.
    """
    stmts = output_json.get("counterfactual_statements", [])
    if not isinstance(stmts, list) or len(stmts) == 0:
        return 0.0
    good = 0
    total = 0
    for s in stmts:
        if not isinstance(s, dict):
            continue
        total += 1
        ev = s.get("evidence", [])
        if not isinstance(ev, list) or len(ev) == 0:
            continue
        dir_out = str(s.get("effect_direction", "unclear")).lower().strip()
        if direction_lookup:
            # Check any cited evidence supports direction
            ok = False
            for e in ev:
                eid = str(e).replace("IR:", "").replace("ENV:", "").replace("LIT:", "").strip()
                info = direction_lookup.get(eid)
                if not info:
                    continue
                if str(info.get("effect_direction", "unclear")).lower().strip() == dir_out and dir_out in ("increase","decrease"):
                    ok = True
                    break
            if ok:
                good += 1
        else:
            # weak fallback: has explicit increase/decrease and evidence
            if dir_out in ("increase","decrease"):
                good += 1
            else:
                # try infer from statement text
                st = str(s.get("statement",""))
                m = _DIR_RE.search(st)
                if m:
                    good += 1
    return float(good / total) if total else 0.0
