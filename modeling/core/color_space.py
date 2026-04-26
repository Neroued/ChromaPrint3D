"""
Project-D65 color-space conversion utilities.

After the color-unification refactor, this module provides **only** the
project-D65 Lab path. The previous OpenCV-Lab helpers (`linear_rgb_to_opencv_lab`,
`linear_rgb_to_opencv_lab_batch`, `lab_jacobian`, `lab_grad_from_linear_batch`)
have been removed; their project-D65 replacements (`linear_rgb_to_lab_d65`,
`lab_d65_jacobian`, `lab_d65_grad_from_linear_batch`) match the C++
`Rgb::ToLab()` / `Lab::DeltaE76` semantics in `core/include/chromaprint3d/color/`.

Key API (post-refactor):
  * `srgb_to_linear` / `linear_to_srgb`              — scalar / array gamma helpers.
  * `linear_rgb_to_xyz`                              — XYZ-D65 forward matrix.
  * `xyz_to_lab(xyz, white)`                         — analytical Lab transform.
  * `linear_rgb_to_lab_d65(linear_rgb)`              — **training-grade** Lab.
                                                       Accepts (3,) or (N,3)
                                                       float arrays; returns
                                                       same-shape float32, NOT
                                                       rounded.
  * `linear_rgb_to_lab_d65_rounded_list(linear_rgb)` — JSON-friendly variant for
                                                       single-pixel writes;
                                                       returns `[L, a, b]`
                                                       rounded to 2 decimals.
  * `lab_d65_jacobian(linear_rgb, eps)`              — 3×3 Jacobian via finite
                                                       diff against
                                                       `linear_rgb_to_lab_d65`.
  * `lab_d65_grad_from_linear_batch(linear_rgb,
                                    dL_dLab, eps)`   — chain-rule gradient
                                                       dLoss/d(linearRGB).

The caller is responsible for sRGB↔linear conversion; we provide the analytic
`linear_rgb_to_lab_d65` as a single source of truth (no OpenCV dependency).
"""

from __future__ import annotations

from typing import List

import numpy as np


# ---------------------------------------------------------------------------
# sRGB  ↔  linear RGB
# ---------------------------------------------------------------------------

def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB [0,1] to linear RGB [0,1]. Element-wise on any-shape array."""
    return np.where(
        rgb <= 0.04045,
        rgb / 12.92,
        ((rgb + 0.055) / 1.055) ** 2.4,
    )


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Convert linear RGB [0,1] to sRGB [0,1]. Element-wise on any-shape array."""
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
    """Linear RGB → XYZ D65. Accepts (3,) or (N,3) float arrays."""
    arr = np.asarray(linear_rgb, dtype=np.float32)
    if arr.ndim == 1:
        return _RGB_TO_XYZ @ arr
    if arr.ndim == 2 and arr.shape[1] == 3:
        # (N,3) batch: apply matrix to each row.
        return arr @ _RGB_TO_XYZ.T
    raise ValueError(f"linear_rgb_to_xyz: expected (3,) or (N,3), got shape {arr.shape}")


def xyz_to_lab(xyz: np.ndarray, white: np.ndarray) -> np.ndarray:
    """XYZ → Lab under given white point. Accepts (3,) or (N,3) arrays."""
    delta = 6.0 / 29.0
    delta_cubed = delta ** 3
    scale = 1.0 / (3.0 * delta * delta)

    xyz_n = xyz / white
    f = np.where(
        xyz_n > delta_cubed,
        np.cbrt(xyz_n),
        xyz_n * scale + 4.0 / 29.0,
    )

    if xyz.ndim == 1:
        l = 116.0 * f[1] - 16.0
        a = 500.0 * (f[0] - f[1])
        b = 200.0 * (f[1] - f[2])
        return np.array([l, a, b], dtype=np.float32)

    # (N,3) batch
    fy = f[:, 1]
    l = 116.0 * fy - 16.0
    a = 500.0 * (f[:, 0] - fy)
    b = 200.0 * (fy - f[:, 2])
    return np.stack([l, a, b], axis=1).astype(np.float32)


def linear_rgb_to_lab_d65(linear_rgb: np.ndarray) -> np.ndarray:
    """Training-grade project-D65 Lab. Accepts (3,) or (N,3) float arrays.

    Returns same-shape float32, **not rounded**. This is the authoritative path
    used by Stage A/B fitting, model-package builders, and the ColorChecker
    calibration pipeline. Matches the C++ `Rgb::ToLab()` definition in
    `core/include/chromaprint3d/color/types.h`.
    """
    arr = np.asarray(linear_rgb, dtype=np.float32)
    arr = np.clip(arr, 0.0, 1.0)
    xyz_d65 = linear_rgb_to_xyz(arr)
    return xyz_to_lab(xyz_d65, _WHITE_D65)


def linear_rgb_to_lab_d65_rounded_list(linear_rgb: np.ndarray) -> List[float]:
    """(3,) → JSON-friendly `[L, a, b]` rounded to 2 decimals.

    Convenience wrapper for single-pixel JSON serialization (Stage A entries,
    color-calibration outputs). Strict 1-D input. For batch / training use
    `linear_rgb_to_lab_d65` directly.
    """
    arr = np.asarray(linear_rgb, dtype=np.float32)
    if arr.ndim != 1 or arr.shape[0] != 3:
        raise ValueError(
            "linear_rgb_to_lab_d65_rounded_list expects a (3,) array; "
            f"got shape {arr.shape}"
        )
    lab = linear_rgb_to_lab_d65(arr)
    return [round(float(v), 2) for v in lab]


# ---------------------------------------------------------------------------
# Lab Jacobian & gradient helpers (project D65 — for Stage B back-prop)
# ---------------------------------------------------------------------------

def lab_d65_jacobian(linear_rgb: np.ndarray, eps: float) -> np.ndarray:
    """3×3 Jacobian dLab/d(linearRGB) under project-D65 Lab via finite diff.

    Replaces the OpenCV-Lab `lab_jacobian` helper from before the refactor.
    Algorithm is identical (central differences clamped to [0,1]); only the
    underlying Lab math changes — `linear_rgb_to_opencv_lab` → `linear_rgb_to_lab_d65`.
    """
    base = np.asarray(linear_rgb, dtype=np.float32)
    if base.ndim != 1 or base.shape[0] != 3:
        raise ValueError(f"lab_d65_jacobian: expected (3,), got shape {base.shape}")
    jac = np.zeros((3, 3), dtype=np.float32)
    for i in range(3):
        plus = base.copy()
        minus = base.copy()
        plus[i] = min(1.0, float(plus[i] + eps))
        minus[i] = max(0.0, float(minus[i] - eps))
        lab_plus = linear_rgb_to_lab_d65(plus)
        lab_minus = linear_rgb_to_lab_d65(minus)
        jac[:, i] = (lab_plus - lab_minus) / (2.0 * eps)
    return jac


def lab_d65_grad_from_linear_batch(
    linear_rgb: np.ndarray,
    dL_dLab: np.ndarray,
    eps: float,
) -> np.ndarray:
    """Chain-rule gradient dLoss/d(linearRGB) given dLoss/dLab (batch).

    Replaces the OpenCV-Lab `lab_grad_from_linear_batch`. Uses
    `linear_rgb_to_lab_d65` batch path internally so the gradient is
    consistent with project-D65 Lab.
    """
    if linear_rgb.ndim != 2 or linear_rgb.shape[1] != 3:
        raise ValueError("lab_d65_grad_from_linear_batch: linear_rgb must be (N,3)")
    if dL_dLab.shape != linear_rgb.shape:
        raise ValueError("lab_d65_grad_from_linear_batch: dL_dLab must match linear_rgb shape")
    dL_dC = np.zeros_like(linear_rgb)
    for i in range(3):
        plus = linear_rgb.copy()
        minus = linear_rgb.copy()
        plus[:, i] = np.minimum(1.0, plus[:, i] + eps)
        minus[:, i] = np.maximum(0.0, minus[:, i] - eps)
        lab_plus = linear_rgb_to_lab_d65(plus)
        lab_minus = linear_rgb_to_lab_d65(minus)
        dLab = (lab_plus - lab_minus) / (2.0 * eps)
        dL_dC[:, i] = np.sum(dL_dLab * dLab, axis=1)
    return dL_dC
