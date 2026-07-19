from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def relative_l2(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.linalg.norm(pred - true) / max(np.linalg.norm(true), 1e-12))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def penetration_fraction(c: np.ndarray, threshold: float = 0.2) -> float:
    return float(np.mean(c >= threshold))


def barrier_auc(conductance: np.ndarray, barrier: np.ndarray) -> float:
    if len(np.unique(barrier)) < 2:
        return float("nan")
    return float(roc_auc_score(barrier, -conductance))
