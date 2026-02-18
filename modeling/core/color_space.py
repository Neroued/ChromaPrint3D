"""
Unified color-space conversion utilities.

Covers sRGB ↔ linear RGB, linear RGB → XYZ → Lab (D65),
and OpenCV-based Lab conversions (single / batch / gradient).
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# sRGB  ↔  linear RGB
# ---------------------------------------------------------------------------

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear RGB [0,1]."""
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB [0,1] to sRGB [0,1]."""
    return np.where(
        rgb <= 0.0031308,
        rgb * 12.92,
        1.055 * np.power(rgb, 1.0 / 2.4) - 0.055,
    )


# ---------------------------------------------------------------------------
# linear RGB  →  XYZ  →  Lab  (analytical, no OpenCV dependency)
# ---------------------------------------------------------------------------

_RGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float32,
)

_WHITE_D65 = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)


def linear_rgb_to_xyz(linear_rgb: np.ndarray) -> np.ndarray:
    """linear RGB (3,) → XYZ D65 (3,)."""
    return _RGB_TO_XYZ @ linear_rgb


def xyz_to_lab(xyz: np.ndarray, white: np.ndarray) -> np.ndarray:
    """XYZ (3,) → Lab (3,) under the given white point."""
    delta = 6.0 / 29.0
    delta_cubed = delta ** 3
    scale = 1.0 / (3.0 * delta * delta)

    xyz_n = xyz / white
    f = np.where(
        xyz_n > delta_cubed,
        np.cbrt(xyz_n),
        xyz_n * scale + 4.0 / 29.0,
    )

    l = 116.0 * f[1] - 16.0
    a = 500.0 * (f[0] - f[1])
    b = 200.0 * (f[1] - f[2])
    return np.array([l, a, b], dtype=np.float32)


def linear_rgb_to_lab_d65(linear_rgb: np.ndarray) -> List[float]:
    """linear RGB (3,) → Lab D65, returned as a rounded list."""
    linear = np.clip(np.asarray(linear_rgb, dtype=np.float32), 0.0, 1.0)
    xyz_d65 = linear_rgb_to_xyz(linear)
    lab = xyz_to_lab(xyz_d65, _WHITE_D65)
    return [round(float(v), 2) for v in lab]


# ---------------------------------------------------------------------------
# OpenCV-backed Lab conversions (single & batch)
# ---------------------------------------------------------------------------

def linear_rgb_to_opencv_lab(linear_rgb: np.ndarray) -> np.ndarray:
    """Single linear RGB (3,) → OpenCV Lab (3,)."""
    linear = np.clip(np.asarray(linear_rgb, dtype=np.float32), 0.0, 1.0)
    srgb = linear_to_srgb(linear)
    bgr = srgb[::-1].reshape(1, 1, 3).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    return lab.reshape(3).astype(np.float32)


def linear_rgb_to_opencv_lab_batch(linear_rgb: np.ndarray) -> np.ndarray:
    """Batch linear RGB (N,3) → OpenCV Lab (N,3)."""
    linear = np.clip(np.asarray(linear_rgb, dtype=np.float32), 0.0, 1.0)
    if linear.ndim != 2 or linear.shape[1] != 3:
        raise ValueError("linear_rgb batch must be shape (N, 3)")
    srgb = linear_to_srgb(linear)
    bgr = srgb[:, ::-1].reshape(-1, 1, 3).astype(np.float32)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    return lab.reshape(-1, 3).astype(np.float32)


# ---------------------------------------------------------------------------
# Lab Jacobian & gradient helpers (for Stage B back-prop)
# ---------------------------------------------------------------------------

def lab_jacobian(linear_rgb: np.ndarray, eps: float) -> np.ndarray:
    """3×3 Jacobian  dLab / d(linearRGB)  via finite differences."""
    base = np.asarray(linear_rgb, dtype=np.float32)
    jac = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        plus = base.copy()
        minus = base.copy()
        plus[i] = min(1.0, float(plus[i] + eps))
        minus[i] = max(0.0, float(minus[i] - eps))
        lab_plus = linear_rgb_to_opencv_lab(plus)
        lab_minus = linear_rgb_to_opencv_lab(minus)
        jac[:, i] = (lab_plus - lab_minus) / (2.0 * eps)
    return jac


def lab_grad_from_linear_batch(
    linear_rgb: np.ndarray,
    dL_dLab: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Chain-rule gradient  dLoss/d(linearRGB)  given  dLoss/dLab  (batch)."""
    if linear_rgb.ndim != 2 or linear_rgb.shape[1] != 3:
        raise ValueError("linear_rgb batch must be shape (N, 3)")
    if dL_dLab.shape != linear_rgb.shape:
        raise ValueError("dL_dLab must match linear_rgb shape")
    dL_dC = np.zeros_like(linear_rgb)
    for i in range(3):
        plus = linear_rgb.copy()
        minus = linear_rgb.copy()
        plus[:, i] = np.minimum(1.0, plus[:, i] + eps)
        minus[:, i] = np.maximum(0.0, minus[:, i] - eps)
        lab_plus = linear_rgb_to_opencv_lab_batch(plus)
        lab_minus = linear_rgb_to_opencv_lab_batch(minus)
        dLab = (lab_plus - lab_minus) / (2.0 * eps)
        dL_dC[:, i] = np.sum(dL_dLab * dLab, axis=1)
    return dL_dC
