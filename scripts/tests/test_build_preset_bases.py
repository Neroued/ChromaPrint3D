"""Unit tests for `scripts/build_preset_bases.py` (plan v13 §9 step 3 verification).

Covers:
  - inherits-chain merge (process / filament / printer)
  - hard-fail on missing template references
  - Step 3.5 K_process pre-alignment for filament_options_with_variant fields
    across (K_filament_raw, K_process) combinations: (1,1), (2,2), (4,4),
    (1,2 - script auto-promotes 1->2), (2,5 H2D pattern), (2,4 H2C pattern)
  - pass-through fields (extruder_variant_list, print_extruder_variant)
  - filament 1-slot user-level field injection
  - EXCLUDED_DYNAMIC_FIELDS removal
  - generated 26 base files match the expected schema

Tests are skipped automatically when BambuStudio clone is not present.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
BAMBU_RESOURCES = REPO_ROOT / "BambuStudio" / "resources" / "profiles"
PRESET_BASES_DIR = REPO_ROOT / "data" / "preset_bases"
MACHINES_PATH = REPO_ROOT / "data" / "presets" / "machines.json"

sys.path.insert(0, str(SCRIPTS_DIR))
import build_preset_bases as bpb  # noqa: E402


def _require_bambu():
    if not BAMBU_RESOURCES.exists():
        raise unittest.SkipTest("BambuStudio clone not present; skipping integration tests")


class FieldSetExtractionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_bambu()
        cls.sets = bpb.load_field_sets(str(BAMBU_RESOURCES))

    def test_print_options_with_variant_known_keys(self) -> None:
        for k in ("outer_wall_speed", "inner_wall_speed", "top_surface_speed",
                  "print_extruder_variant", "print_extruder_id"):
            self.assertIn(k, self.sets["print_options_with_variant"])

    def test_filament_options_with_variant_known_keys(self) -> None:
        for k in ("nozzle_temperature", "nozzle_temperature_initial_layer",
                  "filament_max_volumetric_speed", "filament_extruder_variant"):
            self.assertIn(k, self.sets["filament_options_with_variant"])

    def test_three_sets_disjoint(self) -> None:
        a = self.sets["print_options_with_variant"]
        b = self.sets["filament_options_with_variant"]
        c = self.sets["printer_options_with_variant_1"] | self.sets["printer_options_with_variant_2"]
        self.assertEqual(a & b, set())
        self.assertEqual(a & c, set())
        self.assertEqual(b & c, set())


class BuildBaseDictTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_bambu()
        cls.registry = bpb.BambuRegistry(str(BAMBU_RESOURCES))
        cls.sets = bpb.load_field_sets(str(BAMBU_RESOURCES))
        with open(MACHINES_PATH, "r", encoding="utf-8") as f:
            cls.machines = json.load(f)["machines"]

    def _build(self, machine_name: str, nozzle: str):
        spec = self.machines[machine_name]
        return bpb.build_base_dict(self.registry, machine_name, spec, nozzle, self.sets)

    def test_p2s_n04_two_extruder(self) -> None:
        base, diag = self._build("Bambu Lab P2S", "0.4")
        self.assertEqual(diag["K_process"], 2)
        self.assertEqual(diag["K_filament_raw"], 2)
        self.assertEqual(diag["topology"], "single")
        self.assertEqual(base["print_extruder_variant"],
                         ["Direct Drive Standard", "Direct Drive High Flow"])
        self.assertEqual(len(base["nozzle_temperature"]), 2)
        self.assertEqual(base["filament_extruder_variant"],
                         ["Direct Drive Standard", "Direct Drive High Flow"])
        self.assertEqual(base["printer_model"], "Bambu Lab P2S")

    def test_h2d_n04_dual_kprocess5_kfilament2(self) -> None:
        base, diag = self._build("Bambu Lab H2D", "0.4")
        self.assertEqual(diag["K_process"], 5)
        self.assertEqual(diag["K_filament_raw"], 2)
        self.assertEqual(diag["topology"], "dual")
        self.assertEqual(len(base["nozzle_temperature"]), 5)
        self.assertEqual(len(base["filament_max_volumetric_speed"]), 5)
        # i % 2 pattern: positions 0,2,4 -> [0]; positions 1,3 -> [1]
        speeds = base["filament_max_volumetric_speed"]
        self.assertEqual(speeds[0], speeds[2])
        self.assertEqual(speeds[2], speeds[4])
        self.assertEqual(speeds[1], speeds[3])

    def test_x2d_n04_kprocess_kfilament_match(self) -> None:
        base, diag = self._build("Bambu Lab X2D", "0.4")
        self.assertEqual(diag["K_process"], 4)
        self.assertEqual(diag["K_filament_raw"], 4)
        self.assertEqual(diag["topology"], "dual")
        # Pass-through: 4 distinct speed values for [DD Std, DD HF, Bowden Std, Bowden HF]
        self.assertEqual(len(base["filament_max_volumetric_speed"]), 4)

    def test_a1_n04_single_extruder_kprocess1(self) -> None:
        base, diag = self._build("Bambu Lab A1", "0.4")
        self.assertEqual(diag["K_process"], 1)
        self.assertEqual(diag["K_filament_raw"], 1)
        self.assertEqual(len(base["nozzle_temperature"]), 1)
        self.assertEqual(len(base["print_extruder_variant"]), 1)

    def test_h2c_n04_dual_kprocess4_kfilament2(self) -> None:
        base, diag = self._build("Bambu Lab H2C", "0.4")
        self.assertEqual(diag["K_process"], 4)
        self.assertEqual(diag["K_filament_raw"], 2)
        self.assertEqual(diag["topology"], "dual")
        # K_process is divisor-aligned (4 = 2 * 2), so i % 2 pattern is exact:
        nt = base["nozzle_temperature"]
        self.assertEqual(nt[0], nt[2])
        self.assertEqual(nt[1], nt[3])


class FilamentInjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_bambu()
        cls.registry = bpb.BambuRegistry(str(BAMBU_RESOURCES))
        cls.sets = bpb.load_field_sets(str(BAMBU_RESOURCES))
        with open(MACHINES_PATH, "r", encoding="utf-8") as f:
            cls.machines = json.load(f)["machines"]

    def test_filament_user_level_fields_injected(self) -> None:
        spec = self.machines["Bambu Lab P2S"]
        base, _ = bpb.build_base_dict(
            self.registry, "Bambu Lab P2S", spec, "0.4", self.sets
        )
        self.assertEqual(base["filament_colour"], ["#FFFFFF"])
        self.assertEqual(base["filament_multi_colour"], ["#FFFFFF"])
        self.assertEqual(base["filament_ids"], ["GFA00"])
        self.assertEqual(base["default_filament_colour"], [""])
        self.assertEqual(base["filament_settings_id"], ["Bambu PLA Basic @BBL P2S"])
        self.assertEqual(base["filament_type"], ["PLA"])
        self.assertEqual(base["filament_vendor"], ["Bambu Lab"])

    def test_filament_settings_id_uses_alias(self) -> None:
        # P1S 0.4 aliases to "Bambu PLA Basic @BBL P1S 0.4 nozzle"
        spec = self.machines["Bambu Lab P1S"]
        base, _ = bpb.build_base_dict(
            self.registry, "Bambu Lab P1S", spec, "0.4", self.sets
        )
        self.assertEqual(base["filament_settings_id"], ["Bambu PLA Basic @BBL P1S 0.4 nozzle"])

    def test_excluded_dynamic_fields_stripped(self) -> None:
        spec = self.machines["Bambu Lab P2S"]
        base, _ = bpb.build_base_dict(
            self.registry, "Bambu Lab P2S", spec, "0.4", self.sets
        )
        for k in bpb.EXCLUDED_DYNAMIC_FIELDS:
            self.assertNotIn(k, base, f"excluded field {k} should not be in base_dict")

    def test_runtime_metadata_fields_stripped(self) -> None:
        spec = self.machines["Bambu Lab P2S"]
        base, _ = bpb.build_base_dict(
            self.registry, "Bambu Lab P2S", spec, "0.4", self.sets
        )
        for k in ("inherits", "inherits_group", "different_settings_to_system",
                  "compatible_printers", "print_compatible_printers", "name",
                  "from", "version"):
            self.assertNotIn(k, base, f"runtime field {k} should not be in base_dict")


class HardFailTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        _require_bambu()
        cls.registry = bpb.BambuRegistry(str(BAMBU_RESOURCES))
        cls.sets = bpb.load_field_sets(str(BAMBU_RESOURCES))

    def test_unregistered_process_template_raises(self) -> None:
        spec = {
            "extruder_topology": "single",
            "nozzles": ["0.4"],
            "process_template": {"0.4": "0.08mm Bogus @BBL Nope"},
            "filament_template": {"0.4": "Bambu PLA Basic @BBL P2S"},
            "printer_template": "Bambu Lab P2S {nozzle} nozzle",
        }
        with self.assertRaises(FileNotFoundError):
            bpb.build_base_dict(self.registry, "Test", spec, "0.4", self.sets)

    def test_unregistered_filament_template_raises(self) -> None:
        spec = {
            "extruder_topology": "single",
            "nozzles": ["0.4"],
            "process_template": {"0.4": "0.08mm High Quality @BBL P2S"},
            "filament_template": {"0.4": "Bambu Bogus @BBL Nope"},
            "printer_template": "Bambu Lab P2S {nozzle} nozzle",
        }
        with self.assertRaises(FileNotFoundError):
            bpb.build_base_dict(self.registry, "Test", spec, "0.4", self.sets)

    def test_unregistered_printer_template_raises(self) -> None:
        spec = {
            "extruder_topology": "single",
            "nozzles": ["0.4"],
            "process_template": {"0.4": "0.08mm High Quality @BBL P2S"},
            "filament_template": {"0.4": "Bambu PLA Basic @BBL P2S"},
            "printer_template": "Bambu Lab Bogus {nozzle} nozzle",
        }
        with self.assertRaises(FileNotFoundError):
            bpb.build_base_dict(self.registry, "Test", spec, "0.4", self.sets)


class GeneratedBaseFilesTest(unittest.TestCase):
    """Validate that the 26 committed preset_bases/*.json files conform to the schema."""

    EXPECTED_FILES = {
        f"{slug}_0.08mm_{n}.json"
        for slug in ("bambu_p2s", "bambu_p1p", "bambu_p1s", "bambu_x1c", "bambu_x1",
                     "bambu_x1e", "bambu_a1", "bambu_a1m", "bambu_h2s", "bambu_h2d",
                     "bambu_h2dp", "bambu_h2c", "bambu_x2d")
        for n in ("n02", "n04")
    }

    def test_all_26_files_present(self) -> None:
        if not PRESET_BASES_DIR.exists():
            self.skipTest("preset_bases not yet generated")
        actual = {p.name for p in PRESET_BASES_DIR.iterdir() if p.suffix == ".json"}
        self.assertEqual(actual, self.EXPECTED_FILES)

    def test_each_file_has_required_fields(self) -> None:
        if not PRESET_BASES_DIR.exists():
            self.skipTest("preset_bases not yet generated")
        required = {
            "_chromaprint3d_meta", "print_extruder_variant", "filament_extruder_variant",
            "extruder_variant_list", "printer_model", "filament_settings_id",
            "filament_colour", "filament_multi_colour", "filament_ids",
            "default_filament_colour", "filament_type", "filament_vendor",
            "nozzle_temperature", "nozzle_temperature_initial_layer",
        }
        for p in PRESET_BASES_DIR.glob("*.json"):
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            missing = required - set(d.keys())
            self.assertFalse(missing, f"{p.name} missing required fields: {missing}")

    def test_each_file_filament_extruder_variant_aligned(self) -> None:
        """filament_extruder_variant must equal print_extruder_variant (length K_process)."""
        if not PRESET_BASES_DIR.exists():
            self.skipTest("preset_bases not yet generated")
        for p in PRESET_BASES_DIR.glob("*.json"):
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            self.assertEqual(
                d["filament_extruder_variant"], d["print_extruder_variant"],
                f"{p.name}: filament_extruder_variant != print_extruder_variant"
            )

    def test_each_file_nozzle_temperature_aligned_to_kprocess(self) -> None:
        if not PRESET_BASES_DIR.exists():
            self.skipTest("preset_bases not yet generated")
        for p in PRESET_BASES_DIR.glob("*.json"):
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            K_process = len(d["print_extruder_variant"])
            self.assertEqual(
                len(d["nozzle_temperature"]), K_process,
                f"{p.name}: nozzle_temperature length {len(d['nozzle_temperature'])} != K_process={K_process}"
            )


class SlugifyTest(unittest.TestCase):
    def test_short_map_known_machines(self) -> None:
        self.assertEqual(bpb.slugify_machine("Bambu Lab P2S"), "bambu_p2s")
        self.assertEqual(bpb.slugify_machine("Bambu Lab A1 mini"), "bambu_a1m")
        self.assertEqual(bpb.slugify_machine("Bambu Lab H2D Pro"), "bambu_h2dp")
        self.assertEqual(bpb.slugify_machine("Bambu Lab X1 Carbon"), "bambu_x1c")

    def test_filename_format(self) -> None:
        self.assertEqual(bpb.output_filename("Bambu Lab P2S", "0.4", 0.08),
                         "bambu_p2s_0.08mm_n04.json")
        self.assertEqual(bpb.output_filename("Bambu Lab H2D", "0.2", 0.08),
                         "bambu_h2d_0.08mm_n02.json")


if __name__ == "__main__":
    unittest.main()
