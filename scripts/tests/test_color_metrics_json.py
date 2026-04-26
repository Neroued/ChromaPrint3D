"""Regression tests for `modeling.core.color_calibration.color_metrics_from_linear_rgb`.

Ensures the returned dict is fully JSON-serialisable (no numpy ndarray /
numpy scalar leaks). Caught during the color-unification refactor where
`measured_lab_d65` was previously left as an `np.ndarray`, breaking
`json.dump` in `step1_extract_stages.py`.
"""

from __future__ import annotations

import json
import pathlib
import sys
import unittest

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from modeling.core.color_calibration import color_metrics_from_linear_rgb  # noqa: E402


class ColorMetricsJsonTest(unittest.TestCase):
    def test_returns_json_friendly_types(self) -> None:
        metrics = color_metrics_from_linear_rgb(np.array([0.2, 0.5, 0.8], dtype=np.float32))

        self.assertIsInstance(metrics, dict)
        self.assertSetEqual(
            set(metrics.keys()),
            {"measured_srgb", "measured_lab_d65", "measured_linear_rgb"},
        )

        srgb = metrics["measured_srgb"]
        self.assertIsInstance(srgb, list)
        self.assertEqual(len(srgb), 3)
        for v in srgb:
            self.assertIsInstance(v, int)
            self.assertGreaterEqual(v, 0)
            self.assertLessEqual(v, 255)

        lab = metrics["measured_lab_d65"]
        self.assertIsInstance(lab, list)
        self.assertEqual(len(lab), 3)
        for v in lab:
            self.assertIsInstance(v, float)

        linear = metrics["measured_linear_rgb"]
        self.assertIsInstance(linear, list)
        self.assertEqual(len(linear), 3)
        for v in linear:
            self.assertIsInstance(v, float)

    def test_json_dumps_roundtrip(self) -> None:
        metrics = color_metrics_from_linear_rgb(np.array([0.0, 0.5, 1.0], dtype=np.float32))
        payload = json.dumps({"patch": metrics})
        decoded = json.loads(payload)
        self.assertEqual(decoded["patch"]["measured_srgb"], metrics["measured_srgb"])
        self.assertEqual(decoded["patch"]["measured_lab_d65"], metrics["measured_lab_d65"])

    def test_handles_clamped_input(self) -> None:
        metrics = color_metrics_from_linear_rgb(np.array([-0.1, 0.4, 1.5], dtype=np.float32))
        self.assertEqual(metrics["measured_linear_rgb"][0], 0.0)
        self.assertEqual(metrics["measured_linear_rgb"][2], 1.0)


if __name__ == "__main__":
    unittest.main()
