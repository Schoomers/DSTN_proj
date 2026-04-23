#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""I/O helpers."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)

def write_json(path: str | Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def append_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def write_csv(path: str | Path, df: pd.DataFrame) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)

def safe_get_env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(name)
    return v if v is not None and str(v).strip() != "" else default
