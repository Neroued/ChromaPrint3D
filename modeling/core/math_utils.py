"""
Shared math utilities: activation functions and helpers.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def softplus_grad(x: np.ndarray) -> np.ndarray:
    return sigmoid(x)


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def softplus_inv(x: np.ndarray) -> np.ndarray:
    x = np.maximum(x, 1e-6)
    return np.log(np.expm1(x))


def round_list(values: Iterable[float], ndigits: int) -> List[float]:
    """Round each value and return a plain list."""
    return [round(float(v), ndigits) for v in values]
