#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BLEU-4 and ROUGE-L implementation (lightweight)."""

from __future__ import annotations
import math
from collections import Counter
from typing import List, Tuple

from stage_II.utils.text import simple_tokenize

def _ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(0, max(0, len(tokens)-n+1))]

def bleu4(candidate: str, reference: str, smooth: float = 1e-9) -> float:
    """Compute BLEU-4 with brevity penalty; very small smoothing to avoid log(0)."""
    cand = simple_tokenize(candidate)
    ref = simple_tokenize(reference)
    if not cand or not ref:
        return 0.0
    precisions = []
    weights = [0.25, 0.25, 0.25, 0.25]
    for n in range(1, 5):
        c_ngr = Counter(_ngrams(cand, n))
        r_ngr = Counter(_ngrams(ref, n))
        if sum(c_ngr.values()) == 0:
            precisions.append(0.0)
            continue
        overlap = 0
        for g, c in c_ngr.items():
            overlap += min(c, r_ngr.get(g, 0))
        p_n = overlap / sum(c_ngr.values())
        precisions.append(max(p_n, smooth))

    # brevity penalty
    c_len = len(cand)
    r_len = len(ref)
    if c_len == 0:
        return 0.0
    bp = 1.0 if c_len > r_len else math.exp(1.0 - (r_len / c_len))
    score = bp * math.exp(sum(w * math.log(p) for w, p in zip(weights, precisions)))
    return float(score)

def _lcs_len(a: List[str], b: List[str]) -> int:
    # dynamic programming, O(n*m) - ok for short summaries
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        for j in range(1, m + 1):
            tmp = dp[j]
            if a[i-1] == b[j-1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j-1])
            prev = tmp
    return dp[m]

def rouge_l_f1(candidate: str, reference: str) -> float:
    cand = simple_tokenize(candidate)
    ref = simple_tokenize(reference)
    if not cand or not ref:
        return 0.0
    lcs = _lcs_len(cand, ref)
    prec = lcs / len(cand) if cand else 0.0
    rec = lcs / len(ref) if ref else 0.0
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))
