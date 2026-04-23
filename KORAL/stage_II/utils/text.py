#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lightweight text helpers (no heavy deps)."""

from __future__ import annotations
import re
from typing import List

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:_[A-Za-z0-9]+)?|[%°]+|\S")

def simple_tokenize(text: str) -> List[str]:
    """A simple tokenizer suitable for BLEU/ROUGE in this project."""
    if text is None:
        return []
    text = str(text).strip()
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())
