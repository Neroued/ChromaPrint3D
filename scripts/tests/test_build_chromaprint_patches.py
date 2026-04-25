"""Unit tests for `data/presets/chromaprint_patches.json` and
`scripts/build_chromaprint_patches.py` (plan v13 §9 step 2 verification).

These tests validate the structure of the canonical patches file plus the
classification logic of the helper script. Tests that depend on a local
BambuStudio clone are skipped automatically when the clone is not present.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PATCHES_PATH = REPO_ROOT / "data" / "presets" / "chromaprint_patches.json"
SCRIPTS_DIR = REPO_ROOT / "scripts"
BAMBU_RESOURCES = REPO_ROOT / "BambuStudio" / "resources" / "profiles"

sys.path.insert(0, str(SCRIPTS_DIR))
import build_chromaprint_patches as bcp  # noqa: E402


_VARIANTS = {
    "Direct Drive Standard",
    "Direct Drive High Flow",
    "Direct Drive TPU High Flow",
    "Bowden Standard",
    "Bowden High Flow",
}


class CanonicalPatchesStructureTest(unittest.TestCase):
    """Validate the hand-authored chromaprint_patches.json schema."""

    @classmethod
    def setUpClass(cls) -> None:
        with open(PATCHES_PATH, "r", encoding="utf-8") as f:
            cls.data = json.load(f)

    def test_top_level_sections(self) -> None:
        for key in ("process_common", "process_per_nozzle", "process_per_face", "filament_common"):
            self.assertIn(key, self.data, f"missing top-level section {key}")

    def test_process_per_nozzle_has_supported_nozzles(self) -> None:
        nozzles = self.data["process_per_nozzle"]
        for n in ("0.2", "0.4"):
            self.assertIn(n, nozzles, f"missing nozzle section {n}")

    def test_process_per_face_has_face_up_and_face_down(self) -> None:
        faces = self.data["process_per_face"]
        for f in ("FaceUp", "FaceDown"):
            self.assertIn(f, faces, f"missing face section {f}")

    def test_face_down_overrides_initial_layer_print_height(self) -> None:
        self.assertEqual(
            self.data["process_per_face"]["FaceDown"]["initial_layer_print_height"],
            "${layer_height}",
        )

    def test_variant_indexed_fields_complete(self) -> None:
        """Every $variant_indexed dict must cover all 5 known variants."""
        seen = []
        def walk(node, path):
            if isinstance(node, dict):
                if "$variant_indexed" in node:
                    seen.append(path)
                    keys = set(node["$variant_indexed"].keys())
                    self.assertTrue(
                        _VARIANTS.issubset(keys),
                        f"{path} missing variants: {_VARIANTS - keys}",
                    )
                    for v in node["$variant_indexed"].values():
                        self.assertNotIn("TODO_", v, f"{path} has unresolved TODO placeholder")
                else:
                    for k, v in node.items():
                        walk(v, f"{path}.{k}")
        walk(self.data, "")
        self.assertGreater(len(seen), 0, "no $variant_indexed fields found - schema regression")

    def test_no_filament_modifications(self) -> None:
        """ChromaPrint3D currently does not modify filament_options_with_variant fields."""
        self.assertEqual(self.data["filament_common"], {})


class ScriptClassificationTest(unittest.TestCase):
    """Validate `_classify` labels each known field correctly."""

    @classmethod
    def setUpClass(cls) -> None:
        if not BAMBU_RESOURCES.exists():
            raise unittest.SkipTest("BambuStudio clone not present; skipping classification tests")
        cls.sets = bcp.load_field_sets(
            os.path.join(str(BAMBU_RESOURCES), "..", "..", "src", "libslic3r", "PrintConfig.cpp")
        )

    def test_print_options_with_variant_examples(self) -> None:
        for k in ("outer_wall_speed", "inner_wall_speed", "top_surface_speed",
                  "internal_solid_infill_speed", "sparse_infill_speed",
                  "initial_layer_infill_speed"):
            self.assertEqual(bcp._classify(k, self.sets), "print_with_variant",
                             f"expected print_with_variant for {k!r}")

    def test_filament_options_with_variant_examples(self) -> None:
        for k in ("nozzle_temperature", "nozzle_temperature_initial_layer",
                  "filament_max_volumetric_speed", "filament_flow_ratio",
                  "filament_extruder_variant"):
            self.assertEqual(bcp._classify(k, self.sets), "filament_with_variant",
                             f"expected filament_with_variant for {k!r}")

    def test_printer_extruder_options_examples(self) -> None:
        for k in ("nozzle_diameter", "extruder_type"):
            self.assertEqual(bcp._classify(k, self.sets), "printer_extruder",
                             f"expected printer_extruder for {k!r}")

    def test_printer_options_with_variant_examples(self) -> None:
        for k in ("nozzle_volume", "machine_max_acceleration_x"):
            self.assertEqual(bcp._classify(k, self.sets), "printer_with_variant",
                             f"expected printer_with_variant for {k!r}")

    def test_scalar_default(self) -> None:
        for k in ("wall_generator", "wall_loops", "min_bead_width",
                  "initial_layer_print_height", "line_width"):
            self.assertEqual(bcp._classify(k, self.sets), "scalar",
                             f"expected scalar for {k!r}")

    def test_three_variant_sets_disjoint(self) -> None:
        a = self.sets["print_options_with_variant"]
        b = self.sets["filament_options_with_variant"]
        c = self.sets["printer_options_with_variant_1"] | self.sets["printer_options_with_variant_2"]
        self.assertEqual(a & b, set(), "print_options_with_variant ∩ filament_options_with_variant non-empty")
        self.assertEqual(a & c, set(), "print_options_with_variant ∩ printer_options_with_variant non-empty")
        self.assertEqual(b & c, set(), "filament_options_with_variant ∩ printer_options_with_variant non-empty")


class ToVariantIndexedTest(unittest.TestCase):
    def test_basic_p2s_two_variant(self) -> None:
        out = bcp.to_variant_indexed(
            ["50", "60"],
            ["Direct Drive Standard", "Direct Drive High Flow"],
            extra_variants=bcp._VARIANT_FALLBACKS,
        )
        self.assertEqual(out["Direct Drive Standard"], "50")
        self.assertEqual(out["Direct Drive High Flow"], "60")
        # Variants not in P2S reference get TODO_ placeholders.
        for missing in ("Direct Drive TPU High Flow", "Bowden Standard", "Bowden High Flow"):
            self.assertTrue(out[missing].startswith("TODO_"), f"{missing!r} should be TODO placeholder")

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            bcp.to_variant_indexed(
                ["50", "60", "70"],
                ["Direct Drive Standard", "Direct Drive High Flow"],
                extra_variants=[],
            )


if __name__ == "__main__":
    unittest.main()
