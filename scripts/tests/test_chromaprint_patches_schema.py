"""Schema validation for `data/presets/chromaprint_patches.json`.

The patches file is a hand-maintained artifact (plan v13.1 / m3). The original
draft generator was retired; future maintenance is direct edits to the JSON.
This module locks in the contract BambuStudio relies on:

- top-level sections: `process_common`, `process_per_nozzle`, `process_per_face`, `filament_common`
- `process_per_nozzle` covers `0.2` and `0.4`
- `process_per_face` covers `FaceUp` and `FaceDown`
- every `$variant_indexed` dict covers all 5 known extruder variants and contains no `TODO_*` placeholders
- `filament_common` is empty (ChromaPrint3D does not modify filament_options_with_variant fields)
"""

from __future__ import annotations

import json
import pathlib
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PATCHES_PATH = REPO_ROOT / "data" / "presets" / "chromaprint_patches.json"


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

    # -----------------------------------------------------------------
    # plan v13.2 / chromaprint_patches.json field audit (H2C primary machine)
    # -----------------------------------------------------------------

    def test_color_penetration_layers_set_to_one(self) -> None:
        """Color penetration to all layers (no opaque shell)."""
        self.assertEqual(self.data["process_common"]["bottom_color_penetration_layers"], "1")
        self.assertEqual(self.data["process_common"]["top_color_penetration_layers"], "1")

    def test_shell_layers_minimal(self) -> None:
        """Minimal shells (color layers serve as visible surfaces)."""
        self.assertEqual(self.data["process_common"]["bottom_shell_layers"], "1")
        self.assertEqual(self.data["process_common"]["top_shell_layers"], "0")

    def test_elefant_foot_compensation_zero(self) -> None:
        self.assertEqual(self.data["process_common"]["elefant_foot_compensation"], "0")

    def test_initial_layer_flow_ratio_12(self) -> None:
        """H2C primary machine first-layer flow boost (1.2)."""
        self.assertEqual(self.data["process_common"]["initial_layer_flow_ratio"], "1.2")

    def test_min_bead_width_65pct(self) -> None:
        """H2C primary machine bead width."""
        self.assertEqual(self.data["process_common"]["min_bead_width"], "65%")

    def test_dd_std_speeds_unified_50(self) -> None:
        """All DD Std speeds in process_common are 50 (slow for color edges)."""
        speed_keys = [
            "outer_wall_speed", "inner_wall_speed", "sparse_infill_speed",
            "internal_solid_infill_speed", "top_surface_speed", "initial_layer_infill_speed",
        ]
        for key in speed_keys:
            self.assertIn(key, self.data["process_common"], f"missing {key}")
            entry = self.data["process_common"][key]
            self.assertIn("$variant_indexed", entry, f"{key} not variant-indexed")
            self.assertEqual(
                entry["$variant_indexed"]["Direct Drive Standard"], "50",
                f"{key}.DD Std should be 50, got {entry['$variant_indexed']['Direct Drive Standard']!r}",
            )

    def test_h2c_primary_dd_hf_speeds(self) -> None:
        """H2C primary machine DD HF speeds (process_common)."""
        expected = {
            "outer_wall_speed": "60",
            "inner_wall_speed": "120",
            "sparse_infill_speed": "100",
            "internal_solid_infill_speed": "120",
            "top_surface_speed": "120",
            "initial_layer_infill_speed": "70",
        }
        for key, val in expected.items():
            entry = self.data["process_common"][key]["$variant_indexed"]
            self.assertEqual(
                entry["Direct Drive High Flow"], val,
                f"{key}.DD HF should be {val}, got {entry['Direct Drive High Flow']!r}",
            )

    def test_no_redundant_system_value_overrides(self) -> None:
        """line_width and smooth_coefficient must NOT be in patches (== system defaults)."""
        for section_name in ("process_common",):
            section = self.data[section_name]
            for redundant_key in ("line_width", "smooth_coefficient"):
                self.assertNotIn(
                    redundant_key, section,
                    f"{section_name}.{redundant_key} is redundant (== system); should not be patched",
                )


if __name__ == "__main__":
    unittest.main()
