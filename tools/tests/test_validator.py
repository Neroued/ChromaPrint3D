"""Tests for tools.colordb.validator covering V1-V25 and structural rules.

Positive cases build a minimal valid document with ``_valid()`` and
mutate a single field to trigger each rule. Each rule has at least one
negative test; multi-aspect rules (V20, V23, V24) have several.
"""

from __future__ import annotations

import copy
import json
import struct
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tools.colordb.validator import (
    ColorDBValidationError,
    ValidationReport,
    validate,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _valid() -> Dict[str, Any]:
    """A minimal valid ColorDB document (2 palette entries, 1 section)."""
    return {
        "schema_version": 1,
        "name": "mini",
        "palette": [
            {
                "channel_class": "White",
                "display_name": "White",
                "material": "PLA Basic",
            },
            {
                "channel_class": "Red",
                "display_name": "Red",
                "material": "PLA Basic",
                "hex_color": "#ff0000",
            },
        ],
        "defaults": {
            "color_layers": 3,
            "layer_height_mm": 0.08,
            "line_width_mm": 0.42,
            "base_layers": 10,
            "base_channel_idx": 0,
        },
        "sections": [
            {
                "type": "measured",
                "entries": [
                    {"lab": [93.87, -1.51, 1.32], "recipe": [0, 0, 0]},
                    {"lab": [45.21, 52.33, -8.76], "recipe": [1, 1, 1]},
                ],
            }
        ],
    }


def _rules(report: ValidationReport) -> List[str]:
    return [i.rule for i in report.errors]


def _warn_rules(report: ValidationReport) -> List[str]:
    return [i.rule for i in report.warnings]


class TestBaseline:
    def test_valid_passes(self) -> None:
        assert validate(_valid()).ok

    def test_appendix_a_passes(self) -> None:
        doc = json.loads((FIXTURES / "appendix_a.json").read_text(encoding="utf-8"))
        assert validate(doc).ok


class TestV1:
    def test_missing_schema_version(self) -> None:
        doc = _valid()
        del doc["schema_version"]
        assert "V1" in _rules(validate(doc))

    def test_wrong_schema_version_int(self) -> None:
        doc = _valid()
        doc["schema_version"] = 2
        assert "V1" in _rules(validate(doc))

    def test_string_1_rejected(self) -> None:
        doc = _valid()
        doc["schema_version"] = "1"
        assert "V1" in _rules(validate(doc))


class TestV2:
    def test_empty_palette_rejected(self) -> None:
        doc = _valid()
        doc["palette"] = []
        doc["defaults"]["base_channel_idx"] = 0
        assert "V2" in _rules(validate(doc))

    def test_oversize_palette_rejected(self) -> None:
        doc = _valid()
        # Build a palette of 256 entries with distinct display_names.
        doc["palette"] = [
            {
                "channel_class": "Other",
                "display_name": f"X{i}",
                "material": "m",
            }
            for i in range(256)
        ]
        assert "V2" in _rules(validate(doc))


class TestV3Uniqueness:
    def test_duplicate_normalized_key_rejected(self) -> None:
        doc = _valid()
        doc["palette"] = [
            {"channel_class": "White", "display_name": "White", "material": "PLA Basic"},
            # Same after normalize(): ("white", "plabasic")
            {"channel_class": "White", "display_name": "white", "material": "pla basic"},
        ]
        assert "V3" in _rules(validate(doc))

    def test_chinese_vs_english_names_distinct(self) -> None:
        doc = _valid()
        doc["palette"] = [
            {"channel_class": "Red", "display_name": "Red", "material": "m"},
            {"channel_class": "Red", "display_name": "大红", "material": "m"},
        ]
        assert "V3" not in _rules(validate(doc))


class TestV4V5Defaults:
    def test_missing_defaults_field(self) -> None:
        doc = _valid()
        del doc["defaults"]["layer_height_mm"]
        assert "V4" in _rules(validate(doc))

    def test_wrong_type_defaults_field(self) -> None:
        doc = _valid()
        doc["defaults"]["color_layers"] = "3"
        assert "V4" in _rules(validate(doc))

    # --- B2: int fields MUST reject float literals (even whole ones
    # like 3.0); float fields MUST accept integer literals. ---

    def test_defaults_int_field_rejects_float_literal(self) -> None:
        doc = _valid()
        doc["defaults"]["color_layers"] = 3.0  # float literal for int field
        assert "V4" in _rules(validate(doc))

    def test_defaults_int_field_rejects_bool(self) -> None:
        doc = _valid()
        doc["defaults"]["color_layers"] = True  # bool is not an int per spec
        assert "V4" in _rules(validate(doc))

    def test_defaults_float_field_accepts_int_literal(self) -> None:
        doc = _valid()
        doc["defaults"]["layer_height_mm"] = 1  # int literal for float field
        # Must NOT trigger V4; must NOT trigger V23 (1 > 0).
        rep = validate(doc)
        assert "V4" not in _rules(rep)
        assert "V23" not in _rules(rep)

    def test_v5_base_channel_idx_out_of_range(self) -> None:
        doc = _valid()
        doc["defaults"]["base_channel_idx"] = 10  # palette has 2 entries
        assert "V5" in _rules(validate(doc))


class TestV6V7:
    def test_missing_section_type(self) -> None:
        doc = _valid()
        del doc["sections"][0]["type"]
        assert "V6" in _rules(validate(doc))

    def test_missing_section_entries(self) -> None:
        doc = _valid()
        del doc["sections"][0]["entries"]
        assert "V6" in _rules(validate(doc))

    def test_invalid_type_value(self) -> None:
        doc = _valid()
        doc["sections"][0]["type"] = "simulated"
        assert "V7" in _rules(validate(doc))


class TestV8V12SHOULDs:
    def test_v8_duplicate_identity_key_warns(self) -> None:
        doc = _valid()
        doc["sections"].append(copy.deepcopy(doc["sections"][0]))
        rep = validate(doc)
        assert rep.ok  # SHOULDs do not invalidate
        assert "V8" in _warn_rules(rep)

    def test_v12_duplicate_recipe_warns(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"].append(
            {"lab": [50.0, 0, 0], "recipe": [0, 0, 0]}
        )
        rep = validate(doc)
        assert rep.ok
        assert "V12" in _warn_rules(rep)


class TestV9V10Recipe:
    def test_v9_recipe_length_mismatch(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["recipe"] = [0, 0]  # color_layers=3
        assert "V9" in _rules(validate(doc))

    def test_v10_recipe_index_out_of_range(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["recipe"] = [0, 0, 5]  # palette len 2
        assert "V10" in _rules(validate(doc))

    def test_air_index_255_accepted(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["recipe"] = [0, 0, 255]
        assert validate(doc).ok

    # --- V10 strengthened: uint8 integer range (formerly S-recipe-elem) ---

    def test_v10_negative_element_rejected(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["recipe"] = [-1, 0, 0]
        rules = _rules(validate(doc))
        assert "V10" in rules
        # Must NOT classify under the old S-recipe-elem tag.
        assert "S-recipe-elem" not in rules

    def test_v10_overflow_element_rejected(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["recipe"] = [256, 0, 0]
        assert "V10" in _rules(validate(doc))

    def test_v10_float_element_rejected(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["recipe"] = [0.5, 0, 0]
        assert "V10" in _rules(validate(doc))

    def test_v10_bool_element_rejected(self) -> None:
        # bool is a subclass of int in Python; validator MUST reject.
        doc = _valid()
        doc["sections"][0]["entries"][0]["recipe"] = [True, 0, 0]
        assert "V10" in _rules(validate(doc))


class TestV11ThresholdMargin:
    def test_threshold_on_measured_rejected(self) -> None:
        doc = _valid()
        doc["sections"][0]["threshold"] = 5.0
        assert "V11" in _rules(validate(doc))

    def test_margin_on_measured_rejected(self) -> None:
        doc = _valid()
        doc["sections"][0]["margin"] = 0.7
        assert "V11" in _rules(validate(doc))

    def test_threshold_on_predicted_ok(self) -> None:
        doc = _valid()
        doc["sections"][0]["type"] = "predicted"
        doc["sections"][0]["threshold"] = 5.0
        doc["sections"][0]["margin"] = 0.7
        assert validate(doc).ok


class TestV13UnknownFields:
    def test_unknown_top_level_ignored(self) -> None:
        doc = _valid()
        doc["custom_stat"] = "ignored"
        assert validate(doc).ok

    def test_unknown_palette_entry_ignored(self) -> None:
        doc = _valid()
        doc["palette"][0]["nickname"] = "snow"
        assert validate(doc).ok

    def test_unknown_section_ignored(self) -> None:
        doc = _valid()
        doc["sections"][0]["experimental_flag"] = True
        assert validate(doc).ok

    def test_unknown_entry_ignored(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["note"] = "outlier"
        assert validate(doc).ok


class TestV16V17Palette:
    def test_v16_missing_channel_class(self) -> None:
        doc = _valid()
        del doc["palette"][0]["channel_class"]
        assert "V16" in _rules(validate(doc))

    def test_v16_enum_case_sensitive(self) -> None:
        doc = _valid()
        doc["palette"][0]["channel_class"] = "white"  # lowercase
        assert "V16" in _rules(validate(doc))

    def test_v16_unknown_enum(self) -> None:
        doc = _valid()
        doc["palette"][0]["channel_class"] = "Lemon"
        assert "V16" in _rules(validate(doc))

    def test_v17_missing_display_name(self) -> None:
        doc = _valid()
        del doc["palette"][0]["display_name"]
        assert "V17" in _rules(validate(doc))

    def test_v17_empty_display_name(self) -> None:
        doc = _valid()
        doc["palette"][0]["display_name"] = ""
        assert "V17" in _rules(validate(doc))

    # --- V17 strengthened: after §4.1 normalize MUST retain at least one
    # non-White_Space, non-Default_Ignorable_Code_Point character. ---

    def test_v17_whitespace_only_rejected(self) -> None:
        doc = _valid()
        doc["palette"][0]["display_name"] = "   \t "
        assert "V17" in _rules(validate(doc))

    def test_v17_zwsp_only_rejected(self) -> None:
        doc = _valid()
        doc["palette"][0]["display_name"] = "\u200b\u200c\ufeff"
        assert "V17" in _rules(validate(doc))

    def test_v17_soft_hyphen_only_rejected(self) -> None:
        doc = _valid()
        doc["palette"][0]["display_name"] = "\u00ad\u00ad"
        assert "V17" in _rules(validate(doc))

    def test_v17_variation_selector_only_rejected(self) -> None:
        doc = _valid()
        # VS-1..VS-16 are Default_Ignorable; on their own they render nothing.
        doc["palette"][0]["display_name"] = "\ufe00\ufe0f"
        assert "V17" in _rules(validate(doc))

    def test_v17_name_with_trailing_whitespace_ok(self) -> None:
        # Whitespace around a real letter does NOT trigger V17; normalize
        # strips the spaces and the letter remains.
        doc = _valid()
        doc["palette"][0]["display_name"] = "  White  "
        assert validate(doc).ok

    def test_v17_name_with_embedded_zwsp_ok(self) -> None:
        # Hidden ZWSP between real characters does NOT trigger V17 either;
        # the latin letters are neither whitespace nor Default_Ignorable.
        doc = _valid()
        doc["palette"][0]["display_name"] = "Wh\u200bite"
        # V3 uniqueness key uses normalize() which does NOT strip ZWSP;
        # adjust the second entry to keep the uniqueness key distinct.
        doc["palette"][1]["display_name"] = "Red"
        assert validate(doc).ok


class TestV18V19Localized:
    def test_v18_not_an_object(self) -> None:
        doc = _valid()
        doc["palette"][0]["display_name_localized"] = ["en", "zh-CN"]
        assert "V18" in _rules(validate(doc))

    def test_v18_invalid_bcp47_key(self) -> None:
        doc = _valid()
        doc["palette"][0]["display_name_localized"] = {"en_US": "White"}
        assert "V18" in _rules(validate(doc))

    def test_v18_empty_value(self) -> None:
        doc = _valid()
        doc["palette"][0]["display_name_localized"] = {"en": ""}
        assert "V18" in _rules(validate(doc))

    # --- V18 acceptance: spec §5.1 says loader MUST accept any valid
    # BCP 47 tag, including grandfathered and 4-letter primary. ---

    def test_v18_grandfathered_i_klingon_accepted(self) -> None:
        doc = _valid()
        doc["palette"][0]["display_name_localized"] = {"i-klingon": "tlhIngan"}
        assert "V18" not in _rules(validate(doc))

    def test_v18_grandfathered_en_gb_oed_accepted(self) -> None:
        doc = _valid()
        doc["palette"][0]["display_name_localized"] = {"en-GB-oed": "White"}
        assert "V18" not in _rules(validate(doc))

    def test_v18_4_letter_primary_language_accepted(self) -> None:
        doc = _valid()
        doc["palette"][0]["display_name_localized"] = {"abcd": "Test"}
        assert "V18" not in _rules(validate(doc))

    def test_v19_canonical_key_collision(self) -> None:
        doc = _valid()
        doc["palette"][0]["display_name_localized"] = {
            "en-US": "White",
            "EN-us": "Weiss",  # -> en-US after spec-canonicalize
        }
        assert "V19" in _rules(validate(doc))


class TestV20Lab:
    def test_lab_missing(self) -> None:
        doc = _valid()
        del doc["sections"][0]["entries"][0]["lab"]
        assert "V20" in _rules(validate(doc))

    def test_lab_not_3_elements(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["lab"] = [50.0, 0.0]
        assert "V20" in _rules(validate(doc))

    def test_lab_nan_rejected(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["lab"] = [float("nan"), 0.0, 0.0]
        assert "V20" in _rules(validate(doc))

    def test_lab_infinity_rejected(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["lab"] = [float("inf"), 0.0, 0.0]
        assert "V20" in _rules(validate(doc))

    # --- E1: integer literal in lab array MUST be accepted (promoted
    # to float at the model layer). ---

    def test_lab_integer_literals_accepted(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["lab"] = [95, 0, 0]
        assert validate(doc).ok

    def test_lab_mixed_int_float_accepted(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["lab"] = [95, -1.51, 0]
        assert validate(doc).ok

    # --- E2: reader MUST accept any float precision; writer SHOULD
    # keep original. Validator accepts all finite reals. ---

    def test_lab_high_precision_accepted(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["lab"] = [
            93.87654321012345,
            -1.5123456789,
            1.3e-2,
        ]
        assert validate(doc).ok

    def test_lab_scientific_notation_accepted(self) -> None:
        # JSON permits "1e2" etc.; Python json parses them to float.
        doc = _valid()
        doc["sections"][0]["entries"][0]["lab"] = [1e1, 2e0, -3e-1]
        assert validate(doc).ok


class TestV21EntriesArray:
    def test_entries_non_array_rejected(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"] = None
        assert "V21" in _rules(validate(doc))

    def test_entries_empty_array_allowed(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"] = []
        assert validate(doc).ok

    def test_entries_missing_is_v6_not_v21(self) -> None:
        doc = _valid()
        del doc["sections"][0]["entries"]
        rules = _rules(validate(doc))
        assert "V6" in rules
        assert "V21" not in rules


class TestV22Overrides:
    def test_layer_height_override_must_be_number(self) -> None:
        doc = _valid()
        doc["sections"][0]["layer_height_mm"] = "0.08"
        assert "V22" in _rules(validate(doc))

    def test_color_layers_override_must_be_int(self) -> None:
        doc = _valid()
        doc["sections"][0]["color_layers"] = 3.0
        assert "V22" in _rules(validate(doc))

    def test_int_accepted_for_layer_height(self) -> None:
        # Plan's float field policy: int literals accepted, normalized
        # to float at the model layer. ``layer_height_mm: 0`` would
        # still violate V23 (positivity); use 1 to stay positive.
        doc = _valid()
        doc["sections"][0]["layer_height_mm"] = 1  # int
        rep = validate(doc)
        # V22 must not fire; V23 only fires if <= 0.
        assert "V22" not in _rules(rep)


class TestV23Positivity:
    def test_color_layers_zero_rejected(self) -> None:
        doc = _valid()
        doc["defaults"]["color_layers"] = 0
        doc["sections"][0]["entries"][0]["recipe"] = []
        doc["sections"][0]["entries"][1]["recipe"] = []
        assert "V23" in _rules(validate(doc))

    def test_base_layers_zero_allowed(self) -> None:
        doc = _valid()
        doc["defaults"]["base_layers"] = 0
        assert validate(doc).ok

    def test_base_layers_negative_rejected(self) -> None:
        doc = _valid()
        doc["defaults"]["base_layers"] = -1
        assert "V23" in _rules(validate(doc))

    def test_layer_height_non_positive_rejected(self) -> None:
        doc = _valid()
        doc["defaults"]["layer_height_mm"] = 0.0
        assert "V23" in _rules(validate(doc))

    def test_line_width_non_positive_rejected(self) -> None:
        doc = _valid()
        doc["defaults"]["line_width_mm"] = -0.42
        assert "V23" in _rules(validate(doc))


class TestV24OverrideOutOfRange:
    def test_section_base_channel_idx_overflow(self) -> None:
        doc = _valid()
        doc["sections"][0]["base_channel_idx"] = 99
        assert "V24" in _rules(validate(doc))


class TestV25MsgpackBinLengths:
    """V25 is primarily tested in test_codec_msgpack.py; repeat the
    high-level contract once here so that the rule tests are complete."""

    def test_lab_bin_12_ok(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["lab"] = struct.pack(
            "<fff", 50.0, 0.0, 0.0
        )
        assert validate(doc).ok

    def test_lab_bin_wrong_length(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["lab"] = b"\x00" * 10
        assert "V25" in _rules(validate(doc))

    # --- V20 + V25 interaction: a 12-byte bin with NaN / Inf MUST be
    # caught by the validator (not deferred to the model __post_init__).
    # This is the critical "CLI/loader semantic split" bug fixed here. ---

    def test_lab_bin_nan_triggers_v20(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["lab"] = struct.pack(
            "<fff", float("nan"), 0.0, 0.0
        )
        rules = _rules(validate(doc))
        assert "V20" in rules
        assert "V25" not in rules  # length IS correct

    def test_lab_bin_inf_triggers_v20(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["lab"] = struct.pack(
            "<fff", 50.0, float("inf"), 0.0
        )
        assert "V20" in _rules(validate(doc))

    def test_lab_bin_neg_inf_triggers_v20(self) -> None:
        doc = _valid()
        doc["sections"][0]["entries"][0]["lab"] = struct.pack(
            "<fff", 50.0, 0.0, float("-inf")
        )
        assert "V20" in _rules(validate(doc))


class TestStructuralChecks:
    def test_name_must_be_string(self) -> None:
        doc = _valid()
        doc["name"] = 123
        assert "S-name" in _rules(validate(doc))

    def test_name_empty_string_allowed(self) -> None:
        # Spec §2 does NOT require name to be non-empty; plan is
        # explicitly spec-faithful here.
        doc = _valid()
        doc["name"] = ""
        assert validate(doc).ok

    def test_material_empty_string_allowed(self) -> None:
        # Same for palette[].material per §3.1.
        doc = _valid()
        doc["palette"][0]["material"] = ""
        # V3 might fire if uniqueness key collides; ensure it doesn't.
        doc["palette"][1]["material"] = "nonempty"
        assert validate(doc).ok

    def test_vendor_must_be_string_when_present(self) -> None:
        doc = _valid()
        doc["vendor"] = 42
        assert "S-vendor" in _rules(validate(doc))

    def test_material_type_must_be_string_when_present(self) -> None:
        doc = _valid()
        doc["material_type"] = 42
        assert "S-material_type" in _rules(validate(doc))

    def test_palette_required(self) -> None:
        doc = _valid()
        del doc["palette"]
        assert "S-palette-required" in _rules(validate(doc))

    def test_defaults_required(self) -> None:
        doc = _valid()
        del doc["defaults"]
        assert "S-defaults-required" in _rules(validate(doc))

    def test_sections_required(self) -> None:
        doc = _valid()
        del doc["sections"]
        assert "S-sections-required" in _rules(validate(doc))

    def test_meta_must_be_object(self) -> None:
        doc = _valid()
        doc["meta"] = "not-an-object"
        assert "S-meta" in _rules(validate(doc))

    def test_hex_color_format(self) -> None:
        doc = _valid()
        doc["palette"][1]["hex_color"] = "red"
        assert "S-hex_color" in _rules(validate(doc))

    def test_hex_color_uppercase_accepted(self) -> None:
        doc = _valid()
        doc["palette"][1]["hex_color"] = "#FF00AB"
        assert validate(doc).ok

    def test_threshold_must_be_number(self) -> None:
        doc = _valid()
        doc["sections"][0]["type"] = "predicted"
        doc["sections"][0]["threshold"] = "5.0"
        rules = _rules(validate(doc))
        assert "S-threshold" in rules


class TestValidationReport:
    def test_raise_if_errors(self) -> None:
        doc = _valid()
        doc["schema_version"] = 2
        with pytest.raises(ColorDBValidationError):
            validate(doc).raise_if_errors()

    def test_raise_if_no_errors_is_noop(self) -> None:
        validate(_valid()).raise_if_errors()
