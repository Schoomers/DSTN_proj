#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Literature KG retrieval from a Turtle file."""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import re

try:
    import rdflib
    from rdflib import Graph
except Exception:  # pragma: no cover
    rdflib = None
    Graph = None

@dataclass
class Evidence:
    id: str
    text: str
    source: str

class LiteratureKG:
    def __init__(self, ttl_path: Path):
        self.ttl_path = Path(ttl_path)
        self._g = None

    def load(self) -> None:
        if rdflib is None:
            self._g = None
            return
        g = Graph()
        g.parse(str(self.ttl_path), format="turtle")
        self._g = g

    def available(self) -> bool:
        return self._g is not None

    def retrieve(self, query_terms: List[str], limit: int = 8) -> List[Evidence]:
        """Keyword-ish retrieval: SPARQL if available, otherwise TTL text grep."""
        terms = [t.strip() for t in query_terms if t and t.strip()]
        if not terms:
            return []
        if self._g is None:
            return self._retrieve_by_grep(terms, limit)

        # SPARQL: try to pull any literal containing any term
        evidences: List[Evidence] = []
        seen = set()
        for t in terms:
            q = f"""
            SELECT ?s ?p ?o
            WHERE {{
              ?s ?p ?o .
              FILTER (isLiteral(?o) && CONTAINS(LCASE(STR(?o)), LCASE("{t}")))
            }}
            LIMIT {max(3, limit)}
            """
            try:
                rows = list(self._g.query(q))
            except Exception:
                continue
            for i, (s, p, o) in enumerate(rows):
                txt = str(o)
                if not txt or len(txt) < 20:
                    continue
                key = (str(s), str(p), txt)
                if key in seen:
                    continue
                seen.add(key)
                eid = f"LIT_{len(seen)}"
                evidences.append(Evidence(id=eid, text=txt, source=str(s)))
                if len(evidences) >= limit:
                    return evidences
        return evidences[:limit]

    def _retrieve_by_grep(self, terms: List[str], limit: int) -> List[Evidence]:
        """Fallback: grep raw TTL lines for terms."""
        try:
            ttl = self.ttl_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            return []
        lines = ttl.splitlines()
        evidences = []
        seen = set()
        for ln in lines:
            lnl = ln.lower()
            if any(t.lower() in lnl for t in terms):
                if len(ln.strip()) < 30:
                    continue
                if ln in seen:
                    continue
                seen.add(ln)
                evidences.append(Evidence(id=f"LIT_{len(seen)}", text=ln.strip(), source=str(self.ttl_path.name)))
                if len(evidences) >= limit:
                    break
        return evidences
