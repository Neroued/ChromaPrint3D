"""Unit tests for `scripts/build_preset_bases.py`.

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
        self.assertEqual(diag["K_per_extruder"], 2)
        self.assertEqual(diag["K_filament_raw"], 2)
        self.assertEqual(diag["topology"], "single")
        self.assertEqual(base["print_extruder_variant"],
                         ["Direct Drive Standard", "Direct Drive High Flow"])
        self.assertEqual(len(base["nozzle_temperature"]), 2)
        # filament_extruder_variant uses extruder 0's variants (K_per_extruder).
        self.assertEqual(base["filament_extruder_variant"],
                         ["Direct Drive Standard", "Direct Drive High Flow"])
        self.assertEqual(base["printer_model"], "Bambu Lab P2S")

    def test_h2d_n04_dual_kprocess5_kper_extruder2(self) -> None:
        # H2D K_process=5 (extruder 0 has 2 variants, extruder 1 has 3),
        # but K_per_extruder=2 (extruder 0 only). filament arrays use K_per_extruder.
        base, diag = self._build("Bambu Lab H2D", "0.4")
        self.assertEqual(diag["K_process"], 5)
        self.assertEqual(diag["K_per_extruder"], 2)
        self.assertEqual(diag["K_filament_raw"], 2)
        self.assertEqual(diag["topology"], "dual")
        # nozzle_temperature aligned to K_per_extruder=2 (not K_process=5).
        self.assertEqual(len(base["nozzle_temperature"]), 2)
        self.assertEqual(len(base["filament_max_volumetric_speed"]), 2)
        # base values: fdm_filament_pla volumetric_speed = ['25', '40'].
        speeds = base["filament_max_volumetric_speed"]
        self.assertEqual(speeds, ["25", "40"])
        # filament_extruder_variant uses extruder 0's variants only ([DD Std, DD HF]).
        self.assertEqual(base["filament_extruder_variant"],
                         ["Direct Drive Standard", "Direct Drive High Flow"])
        # print_extruder_variant retains all K_process=5 entries (incl TPU HF).
        self.assertEqual(len(base["print_extruder_variant"]), 5)
        self.assertEqual(base["print_extruder_variant"][4], "Direct Drive TPU High Flow")

    def test_x2d_n04_kper_extruder_truncates_filament(self) -> None:
        # X2D K_process=4, K_per_extruder=2 (extruder 0 = DD Std + DD HF).
        # filament base K_filament_raw=4 truncated to K_per_extruder=2 (extruder 0 only).
        base, diag = self._build("Bambu Lab X2D", "0.4")
        self.assertEqual(diag["K_process"], 4)
        self.assertEqual(diag["K_per_extruder"], 2)
        self.assertEqual(diag["K_filament_raw"], 4)
        self.assertEqual(diag["topology"], "dual")
        # filament_max_volumetric_speed truncated to K_per_extruder=2.
        self.assertEqual(len(base["filament_max_volumetric_speed"]), 2)
        # filament_extruder_variant uses extruder 0's variants only (no Bowden).
        self.assertEqual(base["filament_extruder_variant"],
                         ["Direct Drive Standard", "Direct Drive High Flow"])
        # print_extruder_variant retains all 4 entries incl Bowden.
        self.assertEqual(base["print_extruder_variant"][2], "Bowden Standard")

    def test_a1_n04_single_extruder_kprocess1(self) -> None:
        base, diag = self._build("Bambu Lab A1", "0.4")
        self.assertEqual(diag["K_process"], 1)
        self.assertEqual(diag["K_per_extruder"], 1)
        self.assertEqual(diag["K_filament_raw"], 1)
        self.assertEqual(len(base["nozzle_temperature"]), 1)
        self.assertEqual(len(base["print_extruder_variant"]), 1)
        self.assertEqual(len(base["filament_extruder_variant"]), 1)

    def test_h2c_n04_dual_kprocess4_kper_extruder2(self) -> None:
        # H2C K_process=4 (2 extruders × 2 variants), K_per_extruder=2.
        # filament arrays use K_per_extruder (not K_process).
        base, diag = self._build("Bambu Lab H2C", "0.4")
        self.assertEqual(diag["K_process"], 4)
        self.assertEqual(diag["K_per_extruder"], 2)
        self.assertEqual(diag["K_filament_raw"], 2)
        self.assertEqual(diag["topology"], "dual")
        # nozzle_temperature aligned to K_per_extruder=2.
        nt = base["nozzle_temperature"]
        self.assertEqual(nt, ["220", "220"])
        # filament_max_volumetric_speed at K_per_extruder=2: ['25', '40'] from fdm_filament_pla.
        speeds = base["filament_max_volumetric_speed"]
        self.assertEqual(speeds, ["25", "40"])
        # filament_extruder_variant: extruder 0 only.
        self.assertEqual(base["filament_extruder_variant"],
                         ["Direct Drive Standard", "Direct Drive High Flow"])


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

    def test_each_file_filament_extruder_variant_uses_extruder0(self) -> None:
        """filament_extruder_variant must equal extruder_variant_list[0]'s variants (K_per_extruder)."""
        if not PRESET_BASES_DIR.exists():
            self.skipTest("preset_bases not yet generated")
        for p in PRESET_BASES_DIR.glob("*.json"):
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            ext0_csv = d["extruder_variant_list"][0]
            ext0_variants = [v.strip() for v in ext0_csv.split(",") if v.strip()]
            self.assertEqual(
                d["filament_extruder_variant"], ext0_variants,
                f"{p.name}: filament_extruder_variant != extruder_variant_list[0] split"
            )

    def test_each_file_nozzle_temperature_aligned_to_kper_extruder(self) -> None:
        if not PRESET_BASES_DIR.exists():
            self.skipTest("preset_bases not yet generated")
        for p in PRESET_BASES_DIR.glob("*.json"):
            with open(p, "r", encoding="utf-8") as f:
                d = json.load(f)
            ext0_csv = d["extruder_variant_list"][0]
            K_per_extruder = len([v for v in ext0_csv.split(",") if v.strip()])
            self.assertEqual(
                len(d["nozzle_temperature"]), K_per_extruder,
                f"{p.name}: nozzle_temperature length {len(d['nozzle_temperature'])} "
                f"!= K_per_extruder={K_per_extruder}"
            )


class SlugSchemaTest(unittest.TestCase):
    """Validate `machines.json` slug field is the single source of truth."""

    @classmethod
    def setUpClass(cls) -> None:
        with open(MACHINES_PATH, "r", encoding="utf-8") as f:
            cls.machines_json = json.load(f)

    def test_machines_json_each_entry_has_valid_slug(self) -> None:
        import re
        slug_re = re.compile(r"^[a-z0-9_]+$")
        seen = set()
        for name, spec in self.machines_json["machines"].items():
            self.assertIn("slug", spec, f"{name!r} missing slug")
            slug = spec["slug"]
            self.assertTrue(slug_re.match(slug), f"{name!r} bad slug {slug!r}")
            self.assertNotIn(slug, seen, f"{name!r} duplicate slug {slug!r}")
            seen.add(slug)
        self.assertEqual(len(seen), 13)

    def test_validate_machines_schema_accepts_canonical(self) -> None:
        bpb.validate_machines_schema(self.machines_json)

    def test_validate_machines_schema_rejects_missing_slug(self) -> None:
        bad = json.loads(json.dumps(self.machines_json))
        del bad["machines"]["Bambu Lab P2S"]["slug"]
        with self.assertRaises(ValueError):
            bpb.validate_machines_schema(bad)

    def test_validate_machines_schema_rejects_duplicate_slug(self) -> None:
        bad = json.loads(json.dumps(self.machines_json))
        bad["machines"]["Bambu Lab P1S"]["slug"] = bad["machines"]["Bambu Lab P2S"]["slug"]
        with self.assertRaises(ValueError):
            bpb.validate_machines_schema(bad)

    def test_validate_machines_schema_rejects_invalid_slug_chars(self) -> None:
        bad = json.loads(json.dumps(self.machines_json))
        bad["machines"]["Bambu Lab P2S"]["slug"] = "Bambu-P2S"  # uppercase + dash invalid
        with self.assertRaises(ValueError):
            bpb.validate_machines_schema(bad)

    def test_filename_format(self) -> None:
        self.assertEqual(
            bpb.output_filename("bambu_p2s", "0.4", 0.08), "bambu_p2s_0.08mm_n04.json"
        )
        self.assertEqual(
            bpb.output_filename("bambu_h2d", "0.2", 0.08), "bambu_h2d_0.08mm_n02.json"
        )

    def test_generated_filenames_match_slugs(self) -> None:
        slugs = {spec["slug"] for spec in self.machines_json["machines"].values()}
        actual_prefixes = {
            p.name.rsplit("_0.08mm_", 1)[0]
            for p in PRESET_BASES_DIR.glob("*.json")
        }
        self.assertEqual(actual_prefixes, slugs)


class FilamentNoVariantDriftDetectionTest(unittest.TestCase):
    """Detect upstream BambuStudio drift in `filament_options_no_variant` keyset.

    The C++ runtime `IsFilamentNoVariantKey` (in `core/src/geo/bambu_metadata.cpp`)
    classifies fields by:
      1. NOT in `filament_options_with_variant`
      2. NOT in `print_options_with_variant`
      3. EITHER prefix `filament_*` / `default_filament_*`, OR in a hand-curated
         `kExtra` set (~50 entries).

    If BambuStudio adds new no-variant filament fields without `filament_*` prefix
    and they're not in `kExtra`, ChromaPrint3D will silently miss N-slot expansion
    for those fields. This test extracts the `kExtra` set from the C++ source and
    cross-checks against a reference filament chain, alerting maintainers to sync.
    """

    BAMBU_METADATA_CPP = REPO_ROOT / "core" / "src" / "geo" / "bambu_metadata.cpp"

    @classmethod
    def setUpClass(cls) -> None:
        _require_bambu()
        cls.sets = bpb.load_field_sets(str(BAMBU_RESOURCES))
        # Reference inheritance chain: P2S filament merges all common fields.
        cls.registry = bpb.BambuRegistry(str(BAMBU_RESOURCES))
        chain = cls.registry.chain("filament", "Bambu PLA Basic @BBL P2S")
        cls.merged = bpb.merge_chain(chain)

    @classmethod
    def _extract_cpp_kextra(cls) -> set:
        """Parse `kExtra = { "...", "...", };` from bambu_metadata.cpp."""
        import re
        text = cls.BAMBU_METADATA_CPP.read_text(encoding="utf-8")
        m = re.search(
            r"static const std::unordered_set<std::string>\s+kExtra\s*=\s*\{(.*?)\};",
            text, re.DOTALL,
        )
        if not m:
            raise AssertionError(
                "bambu_metadata.cpp: kExtra set not found - update parser if symbol name changed"
            )
        body = m.group(1)
        return set(re.findall(r'"([^"]+)"', body))

    def test_cpp_kextra_covers_upstream_no_variant_keys(self) -> None:
        """Every non-prefixed no-variant filament field upstream MUST be in C++ kExtra.

        If this fails, BambuStudio added fields that ChromaPrint3D's C++ runtime
        cannot N-slot-expand. Update `kExtra` in `core/src/geo/bambu_metadata.cpp`
        to match.
        """
        kExtra = self._extract_cpp_kextra()
        with_variant = self.sets["filament_options_with_variant"]
        print_variant = self.sets["print_options_with_variant"]

        # Upstream-merged filament chain fields that are NOT prefixed and NOT
        # in the variant sets MUST appear in kExtra.
        # Skip metadata / runtime fields (sourced from script's RUNTIME_METADATA_FIELDS
        # so this test stays in sync as upstream evolves).
        runtime_meta = bpb.RUNTIME_METADATA_FIELDS
        unprefixed_no_variant = {
            k for k in self.merged.keys()
            if not k.startswith("filament_")
            and not k.startswith("default_filament_")
            and k not in with_variant
            and k not in print_variant
            and k not in runtime_meta
            and k != "print_extruder_id"
        }
        missing = unprefixed_no_variant - kExtra
        self.assertEqual(
            missing, set(),
            f"BambuStudio drift: {len(missing)} unprefixed no-variant filament "
            f"field(s) missing from C++ `kExtra`: {sorted(missing)}. "
            f"Update `IsFilamentNoVariantKey` in core/src/geo/bambu_metadata.cpp."
        )


if __name__ == "__main__":
    unittest.main()
