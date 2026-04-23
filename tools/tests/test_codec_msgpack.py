"""Tests for tools.colordb.codec_msgpack (spec §10.4 / V15 / V25)."""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any, Dict

import msgpack  # type: ignore[import-not-found]
import pytest

from tools.colordb import (
    ColorDBValidationError,
    dump_bytes,
    load_bytes,
    load_doc,
)
from tools.colordb.codec_msgpack import (
    decode_msgpack,
    encode_msgpack,
    floats_to_lab_bin,
    lab_bin_to_floats,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _minimal_doc(
    *,
    color_layers: int = 3,
    lab: Any = None,
    recipe: Any = None,
) -> Dict[str, Any]:
    lab_val = [50.0, 0.0, 0.0] if lab is None else lab
    recipe_val = [0, 0, 0] if recipe is None else recipe
    return {
        "schema_version": 1,
        "name": "t",
        "palette": [
            {
                "channel_class": "White",
                "display_name": "W",
                "material": "PLA Basic",
            }
        ],
        "defaults": {
            "color_layers": color_layers,
            "layer_height_mm": 0.08,
            "line_width_mm": 0.42,
            "base_layers": 0,
            "base_channel_idx": 0,
        },
        "sections": [
            {"type": "measured", "entries": [{"lab": lab_val, "recipe": recipe_val}]}
        ],
    }


class TestBinHelpers:
    def test_lab_bin_roundtrip(self) -> None:
        values = (50.0, 10.5, -20.25)
        packed = floats_to_lab_bin(values)
        assert len(packed) == 12
        unpacked = lab_bin_to_floats(packed)
        for a, b in zip(values, unpacked):
            # float32 precision is the best we can promise.
            assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6)

    def test_lab_bin_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError):
            lab_bin_to_floats(b"\x00" * 10)


class TestMsgpackRoundTrip:
    def test_roundtrip_via_high_level_api(self) -> None:
        buf_json = (FIXTURES / "appendix_a.json").read_bytes()
        db = load_bytes(buf_json)
        # Default msgpack dump: recipe -> bin, lab -> array (float64).
        buf_mp = dump_bytes(db, encoding="msgpack")
        db2 = load_bytes(buf_mp)
        for s1, s2 in zip(db.sections, db2.sections):
            assert s1.type == s2.type
            for e1, e2 in zip(s1.entries, s2.entries):
                assert e1.recipe == e2.recipe
                for a, b in zip(e1.lab, e2.lab):
                    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-9)

    def test_msgpack_roundtrip_with_lab_bin(self) -> None:
        buf_json = (FIXTURES / "appendix_a.json").read_bytes()
        db = load_bytes(buf_json)
        buf_mp = dump_bytes(db, encoding="msgpack", lab_as_bin=True)
        # Sanity: bin form carries 12-byte lab for every entry.
        raw = msgpack.unpackb(buf_mp, raw=False, use_list=True)
        for s in raw["sections"]:
            for e in s["entries"]:
                assert isinstance(e["lab"], (bytes, bytearray))
                assert len(e["lab"]) == 12
        db2 = load_bytes(buf_mp)
        assert len(db2.sections) == len(db.sections)

    def test_str_family_used_for_strings(self) -> None:
        # use_bin_type=True => Python str -> str family, not bin family.
        raw = encode_msgpack({"k": "大红"})
        decoded = msgpack.unpackb(raw, raw=False)
        assert decoded == {"k": "大红"}
        # Fixstr header for "k" starts at 0xa1 (fixstr len 1), confirming
        # the str family path is in use.
        assert raw[1] == 0xA1


class TestV15RecipeDualShape:
    def test_accepts_recipe_array(self) -> None:
        doc = _minimal_doc(recipe=[0, 0, 0])
        buf = encode_msgpack(doc)
        db = load_bytes(buf)
        assert db.sections[0].entries[0].recipe == (0, 0, 0)

    def test_accepts_recipe_bin(self) -> None:
        doc = _minimal_doc(recipe=bytes([0, 0, 0]))
        buf = encode_msgpack(doc)
        db = load_bytes(buf)
        assert db.sections[0].entries[0].recipe == (0, 0, 0)

    def test_rejects_recipe_wrong_type(self) -> None:
        # recipe: int is neither array nor bin -> V15
        doc = _minimal_doc()
        doc["sections"][0]["entries"][0]["recipe"] = 42  # type: ignore[index]
        buf = encode_msgpack(doc)
        with pytest.raises(ColorDBValidationError) as excinfo:
            load_bytes(buf)
        rules = {e.rule for e in excinfo.value.report.errors}
        assert "V15" in rules


class TestV25BinLengths:
    def test_accepts_lab_bin_12_bytes(self) -> None:
        doc = _minimal_doc(lab=struct.pack("<fff", 50.0, 0.0, 0.0))
        buf = encode_msgpack(doc)
        db = load_bytes(buf)
        assert db.sections[0].entries[0].lab[0] == pytest.approx(50.0)

    def test_rejects_lab_bin_wrong_length(self) -> None:
        doc = _minimal_doc(lab=b"\x00" * 10)
        buf = encode_msgpack(doc)
        with pytest.raises(ColorDBValidationError) as excinfo:
            load_bytes(buf)
        rules = {e.rule for e in excinfo.value.report.errors}
        assert "V25" in rules

    def test_rejects_recipe_bin_wrong_length(self) -> None:
        # Resolved color_layers=3 but bin length=2
        doc = _minimal_doc(recipe=bytes([0, 0]))
        buf = encode_msgpack(doc)
        with pytest.raises(ColorDBValidationError) as excinfo:
            load_bytes(buf)
        rules = {e.rule for e in excinfo.value.report.errors}
        # V9 catches the length mismatch first; V25 would fire in the
        # scenario where color_layers is known but bin length differs.
        assert "V9" in rules or "V25" in rules
