#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predictive (classification/regression) metrics."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

@dataclass
class Confusion:
    tp: int
    fp: int
    fn: int
    tn: int

    def precision(self) -> float:
        denom = self.tp + self.fp
        return float(self.tp / denom) if denom else 0.0

    def recall(self) -> float:
        denom = self.tp + self.fn
        return float(self.tp / denom) if denom else 0.0

    def accuracy(self) -> float:
        denom = self.tp + self.fp + self.fn + self.tn
        return float((self.tp + self.tn) / denom) if denom else 0.0

def confusion_from_labels(y_true: List[int], y_pred: List[int]) -> Confusion:
    tp = fp = fn = tn = 0
    for yt, yp in zip(y_true, y_pred):
        yt = int(yt)
        yp = int(yp)
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1
        else:
            tn += 1
    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)

def mse(y_true: List[float], y_pred: List[float]) -> float:
    a = np.asarray(y_true, dtype=float)
    b = np.asarray(y_pred, dtype=float)
    if a.size == 0:
        return float("nan")
    return float(np.mean((a - b) ** 2))
