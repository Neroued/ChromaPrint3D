"""Tests for tools.colordb.merge (spec §4.3 palette-only fail-hard)."""

from __future__ import annotations

import pytest

from tools.colordb import ColorDBMergeError, PaletteEntry, merge_palettes


def _e(
    *,
    channel_class: str = "Red",
    display_name: str = "Red",
    material: str = "PLA Basic",
    hex_color: str | None = "#ff0000",
    display_name_localized=None,
) -> PaletteEntry:
    return PaletteEntry(
        channel_class=channel_class,
        display_name=display_name,
        material=material,
        hex_color=hex_color,
        display_name_localized=display_name_localized or {},
    )


class TestHardConflicts:
    def test_channel_class_conflict(self) -> None:
        a = _e(channel_class="Red")
        b = _e(channel_class="Pink")
        with pytest.raises(ColorDBMergeError) as excinfo:
            merge_palettes([[a], [b]])
        assert "channel_class" in str(excinfo.value)

    def test_hex_color_conflict(self) -> None:
        a = _e(hex_color="#ff0000")
        b = _e(hex_color="#ff0001")
        with pytest.raises(ColorDBMergeError) as excinfo:
            merge_palettes([[a], [b]])
        assert "hex_color" in str(excinfo.value)

    def test_display_name_literal_conflict(self) -> None:
        # Same merge key (Red, PLA Basic after normalize), different
        # literal display_name that still normalizes the same.
        a = _e(display_name="Red")
        b = _e(display_name="RED")
        with pytest.raises(ColorDBMergeError) as excinfo:
            merge_palettes([[a], [b]])
        assert "display_name" in str(excinfo.value)

    def test_material_literal_conflict(self) -> None:
        # Normalized: both ("red", "plabasic"); literal differs.
        a = _e(material="PLA Basic")
        b = _e(material="pla basic")
        with pytest.raises(ColorDBMergeError) as excinfo:
            merge_palettes([[a], [b]])
        assert "material" in str(excinfo.value)

    def test_localized_same_key_different_value_conflict(self) -> None:
        a = _e(display_name_localized={"zh-CN": "红"})
        b = _e(display_name_localized={"zh-CN": "火红"})
        with pytest.raises(ColorDBMergeError) as excinfo:
            merge_palettes([[a], [b]])
        assert "zh-CN" in str(excinfo.value)


class TestPositiveMerges:
    def test_localized_complementary_union(self) -> None:
        a = _e(display_name_localized={"zh-CN": "红"})
        b = _e(display_name_localized={"ja-JP": "赤"})
        merged = merge_palettes([[a], [b]])
        assert len(merged) == 1
        loc = dict(merged[0].display_name_localized)
        assert loc == {"zh-CN": "红", "ja-JP": "赤"}

    def test_exact_duplicate_deduped(self) -> None:
        a = _e(display_name_localized={"zh-CN": "红"})
        b = _e(display_name_localized={"zh-CN": "红"})
        merged = merge_palettes([[a], [b]])
        assert len(merged) == 1

    def test_disjoint_entries_preserve_order(self) -> None:
        a = _e(channel_class="Red", display_name="Red")
        b = _e(
            channel_class="White",
            display_name="White",
            hex_color="#ffffff",
        )
        merged = merge_palettes([[a], [b]])
        assert [e.display_name for e in merged] == ["Red", "White"]


class TestCanonicalLocalizedKeys:
    def test_same_canonical_literal_ok(self) -> None:
        # "en-US" and "EN-us" both canonicalize to "en-US"; if the
        # values are equal this is effectively a single entry.
        a = _e(display_name_localized={"en-US": "Red"})
        b = _e(display_name_localized={"EN-us": "Red"})
        merged = merge_palettes([[a], [b]])
        assert len(merged) == 1
        loc = dict(merged[0].display_name_localized)
        assert loc == {"en-US": "Red"}
