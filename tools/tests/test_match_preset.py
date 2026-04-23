"""Tests for §3.5 multiset preset matching."""

from __future__ import annotations

import pytest

from tools.colordb import match_preset


def _as_entries(classes):
    return [{"channel_class": c, "display_name": c, "material": "m"} for c in classes]


class TestExactMultisetMatches:
    def test_rybw(self) -> None:
        assert match_preset(_as_entries(["Red", "Yellow", "Blue", "White"])) == {
            "RYBW",
            "all",
        }

    def test_cmyk(self) -> None:
        assert match_preset(_as_entries(["Cyan", "Magenta", "Yellow", "Black"])) == {
            "CMYK",
            "all",
        }

    def test_cmyw(self) -> None:
        assert match_preset(_as_entries(["Cyan", "Magenta", "Yellow", "White"])) == {
            "CMYW",
            "all",
        }

    def test_cmykw(self) -> None:
        assert match_preset(
            _as_entries(["Cyan", "Magenta", "Yellow", "Black", "White"])
        ) == {"CMYKW", "all"}

    def test_rgb(self) -> None:
        assert match_preset(_as_entries(["Red", "Green", "Blue"])) == {"RGB", "all"}

    def test_rgbw(self) -> None:
        assert match_preset(_as_entries(["Red", "Green", "Blue", "White"])) == {
            "RGBW",
            "all",
        }

    def test_ryb(self) -> None:
        assert match_preset(_as_entries(["Red", "Yellow", "Blue"])) == {"RYB", "all"}

    def test_order_independent(self) -> None:
        assert match_preset(_as_entries(["White", "Blue", "Yellow", "Red"])) == {
            "RYBW",
            "all",
        }


class TestMultisetNegatives:
    def test_red_duplicated_does_not_match_rybw(self) -> None:
        # Spec §3.5 example: {Red x2, Yellow, Blue, White} does NOT
        # match RYBW because Red count differs.
        result = match_preset(_as_entries(["Red", "Red", "Yellow", "Blue", "White"]))
        assert result == {"all"}

    def test_extra_entry_does_not_match(self) -> None:
        result = match_preset(
            _as_entries(["Red", "Yellow", "Blue", "White", "Cyan"])
        )
        assert result == {"all"}

    def test_missing_entry_does_not_match(self) -> None:
        result = match_preset(_as_entries(["Red", "Yellow", "Blue"]))
        # 3 entries -> RYB (positive), definitely NOT RYBW.
        assert "RYBW" not in result
        assert "RYB" in result


class TestNoColorFamilyGeneralization:
    def test_light_red_does_not_substitute_red(self) -> None:
        result = match_preset(_as_entries(["LightRed", "Yellow", "Blue", "White"]))
        assert result == {"all"}

    def test_pink_does_not_substitute_red(self) -> None:
        result = match_preset(_as_entries(["Pink", "Yellow", "Blue", "White"]))
        assert result == {"all"}

    def test_burgundy_does_not_substitute_red(self) -> None:
        result = match_preset(_as_entries(["Burgundy", "Yellow", "Blue", "White"]))
        assert result == {"all"}

    def test_dark_green_does_not_substitute_green(self) -> None:
        result = match_preset(_as_entries(["Red", "DarkGreen", "Blue"]))
        assert result == {"all"}


class TestSpecialAppearanceOtherHitOnlyAll:
    def test_metallic_only_hits_all(self) -> None:
        assert match_preset(_as_entries(["Gold", "Silver", "Bronze"])) == {"all"}

    def test_fluorescent_only_hits_all(self) -> None:
        assert match_preset(_as_entries(["Fluorescent"])) == {"all"}

    def test_other_only_hits_all(self) -> None:
        assert match_preset(_as_entries(["Other", "Other"])) == {"all"}


class TestInputVariants:
    def test_accepts_strings(self) -> None:
        assert match_preset(["Red", "Yellow", "Blue", "White"]) == {"RYBW", "all"}

    def test_accepts_mapping(self) -> None:
        assert match_preset([{"channel_class": "Red"}]) == {"all"}

    def test_empty_palette_hits_all(self) -> None:
        # "all" is always available; empty palette is obviously not a
        # named preset match.
        assert match_preset([]) == {"all"}
