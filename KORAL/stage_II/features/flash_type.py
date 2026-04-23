#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flash type parsing."""

from __future__ import annotations
from typing import Any, Dict

def build_flash_type_ir(row: Dict[str, Any]) -> Dict[str, Any]:
    ft = row.get("flash_type") or row.get("ft") or row.get("FlashType")
    if ft is None:
        return {}
    s = str(ft).strip()
    if not s:
        return {}
    return {"flash_type": {"id": "FT_1", "type": s}}
