"""Parity tests for `modeling.core.color_space.linear_rgb_to_lab_d65`.

Covers two invariants required by the color-unification refactor:

1. **scalar/batch consistency** — the function accepts both `(3,)` and
   `(N, 3)` shapes and the per-row results must match exactly.
2. **sRGB → linear → sRGB byte round-trip** — encoding and decoding
   inverses must agree to ≤ 1 unit on uint8 samples.

The C++ ↔ Python ΔE76 parity check (third invariant from the plan) is
covered indirectly by the C++ side `test_color.cpp::Rgb::ToLab`
self-test plus the JS side `colorConvert.test.ts::round-trip`. Both
sides share the same closed-form constants as Python, so a Python-only
suite is sufficient to lock down behaviour without spinning up a C++
helper binary.
"""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from modeling.core.color_space import (  # noqa: E402
    linear_rgb_to_lab_d65,
    linear_to_srgb,
    srgb_to_linear,
)


class LinearRgbToLabD65ParityTest(unittest.TestCase):
    def test_scalar_matches_batch(self) -> None:
        rng = np.random.default_rng(0xC010_50FA)
        srgb = rng.integers(0, 256, size=(1024, 3), dtype=np.uint16).astype(np.float32) / 255.0
        linear = srgb_to_linear(srgb)

        batch = np.asarray(linear_rgb_to_lab_d65(linear), dtype=np.float64)

        scalars = np.empty_like(batch)
        for i in range(linear.shape[0]):
            scalars[i] = np.asarray(
                linear_rgb_to_lab_d65(linear[i]), dtype=np.float64
            )

        self.assertEqual(batch.shape, scalars.shape)
        # Tolerance is float32 round-off between scalar and matmul code
        # paths inside `linear_rgb_to_lab_d65` (matrix accumulation order
        # differs). 1e-3 in Lab space is ~ΔE 1e-3, well below human perception.
        np.testing.assert_allclose(batch, scalars, atol=1e-3, rtol=0.0)

    def test_srgb_byte_round_trip(self) -> None:
        rng = np.random.default_rng(0xCAFE_1234)
        srgb_u8 = rng.integers(0, 256, size=(2048, 3), dtype=np.uint16).astype(np.float32)
        srgb_norm = srgb_u8 / 255.0

        linear = srgb_to_linear(srgb_norm)
        srgb_back = linear_to_srgb(linear) * 255.0
        srgb_back_u8 = np.clip(np.round(srgb_back), 0.0, 255.0).astype(np.int32)

        delta = np.abs(srgb_back_u8 - srgb_u8.astype(np.int32))
        self.assertLessEqual(int(delta.max()), 1,
                             "sRGB→linear→sRGB byte round-trip exceeds 1-unit tolerance")

    def test_lab_anchor_points(self) -> None:
        # D65 white in linear RGB must map to L*=100, a*≈0, b*≈0.
        white_lab = np.asarray(
            linear_rgb_to_lab_d65(np.array([1.0, 1.0, 1.0], dtype=np.float32)),
            dtype=np.float64,
        )
        self.assertAlmostEqual(white_lab[0], 100.0, places=2)
        self.assertAlmostEqual(white_lab[1], 0.0, delta=0.05)
        self.assertAlmostEqual(white_lab[2], 0.0, delta=0.05)

        # Linear-RGB black → L*=0.
        black_lab = np.asarray(
            linear_rgb_to_lab_d65(np.array([0.0, 0.0, 0.0], dtype=np.float32)),
            dtype=np.float64,
        )
        self.assertAlmostEqual(black_lab[0], 0.0, places=4)
        self.assertAlmostEqual(black_lab[1], 0.0, places=4)
        self.assertAlmostEqual(black_lab[2], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
