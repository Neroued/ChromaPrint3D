"""Tests for tools.colordb.normalize (spec §4.1)."""

from __future__ import annotations

import pytest

from tools.colordb.normalize import normalize


class TestCaseFolding:
    def test_german_sharp_s_folds_to_ss(self) -> None:
        # Unicode Default Case Folding; str.casefold() is correct here.
        # str.lower() is NOT sufficient: "ß".lower() == "ß".
        assert normalize("ß") == "ss"
        assert normalize("Straße") == "strasse"

    def test_ascii_lowercase(self) -> None:
        assert normalize("PLA Basic") == "plabasic"
        assert normalize("Red") == "red"

    def test_mixed_case_folds(self) -> None:
        assert normalize("ABC-def-123") == "abc-def-123"


class TestWhiteSpaceStripping:
    def test_ascii_whitespace_stripped(self) -> None:
        assert normalize("  red  ") == "red"
        assert normalize("\tred\n") == "red"
        assert normalize("r\te\nd") == "red"

    def test_full_width_space_u3000_stripped(self) -> None:
        assert normalize("红\u3000色") == "红色"

    def test_nbsp_stripped(self) -> None:
        # U+00A0 NO-BREAK SPACE is a White_Space codepoint.
        assert normalize("a\u00a0b") == "ab"

    def test_all_25_whitespace_codepoints_stripped(self) -> None:
        white = "".join(
            chr(cp)
            for cp in (
                0x0009, 0x000A, 0x000B, 0x000C, 0x000D,
                0x0020, 0x0085, 0x00A0, 0x1680,
                0x2000, 0x2001, 0x2002, 0x2003, 0x2004,
                0x2005, 0x2006, 0x2007, 0x2008, 0x2009, 0x200A,
                0x2028, 0x2029, 0x202F, 0x205F, 0x3000,
            )
        )
        assert normalize("x" + white + "y") == "xy"


class TestInvisibleRetention:
    @pytest.mark.parametrize(
        "cp",
        [
            0x200B,  # ZERO WIDTH SPACE (NOT White_Space)
            0x200C,  # ZERO WIDTH NON-JOINER
            0x200D,  # ZERO WIDTH JOINER
            0x00AD,  # SOFT HYPHEN
            0xFEFF,  # ZERO WIDTH NO-BREAK SPACE (BOM)
        ],
    )
    def test_invisible_non_whitespace_retained(self, cp: int) -> None:
        ch = chr(cp)
        out = normalize(f"a{ch}b")
        # The character must still be present (normalize is not a
        # "strip all invisibles" function; the spec is explicit).
        assert ch in out
        assert out.startswith("a") and out.endswith("b")


class TestUnicodeContent:
    def test_chinese_characters_preserved(self) -> None:
        assert normalize("大红") == "大红"
        assert normalize("红 色") == "红色"
        # Punctuation survives.
        assert normalize("大-红") == "大-红"

    def test_da_hong_red_distinct(self) -> None:
        assert normalize("大红") != normalize("Red")
        assert normalize("Red") == normalize("red")

    def test_nfc_canonicalization(self) -> None:
        # "é" composed (U+00E9) vs decomposed (U+0065 U+0301)
        composed = "\u00e9"
        decomposed = "e\u0301"
        assert normalize(composed) == normalize(decomposed)


class TestNoNfkc:
    def test_half_width_vs_full_width_remain_distinct(self) -> None:
        # Spec §4.1: "normalize does NOT do NFKC".
        half = "\uff71"  # HALFWIDTH KATAKANA LETTER A -> "ｱ"
        full = "\u30a2"  # KATAKANA LETTER A -> "ア"
        assert normalize(half) != normalize(full)


class TestArgumentTypes:
    def test_rejects_bytes(self) -> None:
        with pytest.raises(TypeError):
            normalize(b"abc")  # type: ignore[arg-type]

    def test_rejects_none(self) -> None:
        with pytest.raises(TypeError):
            normalize(None)  # type: ignore[arg-type]
