"""Tests for the dataclass model and round-trip semantics."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tools.colordb import (
    ChannelClass,
    ColorDB,
    Defaults,
    Entry,
    PaletteEntry,
    ResolvedConfig,
    Section,
    dump_bytes,
    load_bytes,
)

FIXTURES = Path(__file__).parent / "fixtures"


class TestChannelClassEnum:
    def test_has_52_members(self) -> None:
        assert len(ChannelClass) == 52

    def test_str_equality(self) -> None:
        assert ChannelClass.Red == "Red"
        assert ChannelClass("Red") is ChannelClass.Red


class TestResolvedConfig:
    def test_section_inherits_all_from_defaults(self) -> None:
        defaults = Defaults(
            color_layers=5,
            layer_height_mm=0.08,
            line_width_mm=0.42,
            base_layers=10,
            base_channel_idx=0,
        )
        section = Section(type="measured")
        rc = section.resolved_config(defaults)
        assert rc == ResolvedConfig(5, 0.08, 0.42, 10, 0)

    def test_section_overrides_take_precedence(self) -> None:
        defaults = Defaults(
            color_layers=5,
            layer_height_mm=0.08,
            line_width_mm=0.42,
            base_layers=10,
            base_channel_idx=0,
        )
        section = Section(
            type="measured",
            color_layers=3,
            layer_height_mm=0.12,
            base_layers=7,
        )
        rc = section.resolved_config(defaults)
        assert rc.color_layers == 3
        assert rc.layer_height_mm == 0.12
        assert rc.base_layers == 7
        # Non-overridden inherit:
        assert rc.line_width_mm == 0.42
        assert rc.base_channel_idx == 0


class TestEntryValidation:
    def test_rejects_non_finite_lab(self) -> None:
        with pytest.raises(ValueError):
            Entry(lab=(float("nan"), 0.0, 0.0), recipe=(0,))
        with pytest.raises(ValueError):
            Entry(lab=(0.0, float("inf"), 0.0), recipe=(0,))

    def test_rejects_non_3_lab(self) -> None:
        with pytest.raises(ValueError):
            Entry(lab=(0.0, 0.0), recipe=(0,))  # type: ignore[arg-type]


class TestRoundTripAppendixA:
    def test_semantic_roundtrip_json(self) -> None:
        buf = (FIXTURES / "appendix_a.json").read_bytes()
        db1 = load_bytes(buf)
        # JSON dump then re-load must yield an equivalent object.
        again = load_bytes(dump_bytes(db1, encoding="json"))
        assert again.schema_version == db1.schema_version
        assert again.name == db1.name
        assert len(again.palette) == len(db1.palette)
        for a, b in zip(again.palette, db1.palette):
            assert a.channel_class == b.channel_class
            assert a.display_name == b.display_name
            assert a.material == b.material
            assert a.hex_color == b.hex_color
            assert dict(a.display_name_localized) == dict(b.display_name_localized)
        assert again.defaults == db1.defaults
        assert len(again.sections) == len(db1.sections)
        for sa, sb in zip(again.sections, db1.sections):
            assert sa.type == sb.type
            assert len(sa.entries) == len(sb.entries)
            for ea, eb in zip(sa.entries, sb.entries):
                assert eb.recipe == ea.recipe
                for va, vb in zip(ea.lab, eb.lab):
                    assert math.isclose(va, vb, rel_tol=0, abs_tol=1e-9)

    def test_unknown_fields_dropped_by_roundtrip(self) -> None:
        # V13: loader MUST ignore unknown top-level / entry / section
        # fields. A round-trip therefore drops them (semantic equality
        # is preserved even if byte-identity is not).
        import json

        doc = json.loads((FIXTURES / "appendix_a.json").read_text(encoding="utf-8"))
        doc["extra_top"] = "ignored"
        doc["palette"][0]["extra_palette_field"] = "ignored"
        doc["sections"][0]["extra_section_field"] = "ignored"
        doc["sections"][0]["entries"][0]["extra_entry_field"] = "ignored"

        buf = json.dumps(doc).encode("utf-8")
        db = load_bytes(buf)
        dumped = dump_bytes(db, encoding="json")
        out = json.loads(dumped)
        assert "extra_top" not in out
        assert "extra_palette_field" not in out["palette"][0]
        assert "extra_section_field" not in out["sections"][0]
        assert "extra_entry_field" not in out["sections"][0]["entries"][0]
