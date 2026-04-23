"""Tests for spec §5.1.1 spec_canonicalize + is_valid_bcp47."""

from __future__ import annotations

import pytest

from tools.colordb.locale_tag import is_valid_bcp47, spec_canonicalize


class TestSpecCanonicalize:
    # --------------- primary language / extlang ---------------
    def test_primary_language_lowercased(self) -> None:
        assert spec_canonicalize("EN") == "en"
        assert spec_canonicalize("En") == "en"
        assert spec_canonicalize("zh") == "zh"

    def test_extlang_lowercased(self) -> None:
        # primary + 1 extlang
        assert spec_canonicalize("ZH-CMN") == "zh-cmn"
        # primary + extlang + script + region, all with mixed case
        assert spec_canonicalize("ZH-CMN-hanS-cn") == "zh-cmn-Hans-CN"

    # --------------- script -> Title Case ---------------
    def test_script_title_case(self) -> None:
        assert spec_canonicalize("zh-HANT") == "zh-Hant"
        assert spec_canonicalize("zh-hant") == "zh-Hant"
        assert spec_canonicalize("sr-LATN") == "sr-Latn"
        assert spec_canonicalize("ru-cyRL") == "ru-Cyrl"

    # --------------- region ---------------
    def test_alpha_region_uppercased(self) -> None:
        assert spec_canonicalize("en-us") == "en-US"
        assert spec_canonicalize("zh-tw") == "zh-TW"
        assert spec_canonicalize("EN-US") == "en-US"

    def test_digit_region_preserved(self) -> None:
        assert spec_canonicalize("es-419") == "es-419"
        # "es-Latn-419" -> script + region
        assert spec_canonicalize("es-LATN-419") == "es-Latn-419"

    # --------------- variant ---------------
    def test_variant_lowercased(self) -> None:
        assert spec_canonicalize("sl-ROZAJ") == "sl-rozaj"
        assert spec_canonicalize("sl-Latn-IT-NEDIS") == "sl-Latn-IT-nedis"
        # 4-digit-starting variant
        assert spec_canonicalize("de-CH-1996") == "de-CH-1996"

    # --------------- extension ---------------
    def test_extension_lowercased_full(self) -> None:
        assert spec_canonicalize("EN-U-CA-GREGORY") == "en-u-ca-gregory"
        # Region present first then extension
        assert spec_canonicalize("EN-US-U-CA-GREGORY") == "en-US-u-ca-gregory"

    # --------------- privateuse ---------------
    def test_privateuse_lowercased(self) -> None:
        assert spec_canonicalize("en-X-FOO") == "en-x-foo"
        # privateuse-only: spec treats 'x' as primary's position, which we
        # still lowercase; grammar validation is separate.
        assert spec_canonicalize("X-PrivateTag") == "x-privatetag"

    # --------------- no substitution / alias ---------------
    def test_no_substitution_iw_he(self) -> None:
        # Spec §5.1.1: deprecated mappings MUST NOT be applied.
        assert spec_canonicalize("IW") == "iw"
        assert spec_canonicalize("iw-IL") == "iw-IL"

    def test_no_grandfathered_expansion(self) -> None:
        # "i-klingon" stays a malformed-by-position string; we still
        # lowercase everything without expanding.
        # Note: "i" is at position 0 so gets lowercased as "language",
        # which happens to already be lower. The key is that the tag
        # is NOT replaced with "tlh".
        assert spec_canonicalize("i-klingon") == "i-klingon"

    def test_no_likely_subtags(self) -> None:
        # zh-TW stays zh-TW; NOT zh-Hant-TW.
        assert spec_canonicalize("zh-TW") == "zh-TW"
        assert spec_canonicalize("en") == "en"  # not en-Latn-US


class TestIsValidBcp47:
    # --------------- positive cases ---------------
    @pytest.mark.parametrize(
        "tag",
        [
            "en",
            "zh",
            "en-US",
            "zh-Hant",
            "zh-Hant-TW",
            "zh-cmn-Hans-CN",
            "sl-Latn-IT-nedis",
            "de-CH-1996",
            "es-419",
            "en-u-ca-gregory",
            "en-US-u-ca-gregory-nu-latn",
            "en-x-foo",
            "x-private",
            # 4-letter primary language (reserved but syntactically valid
            # per BCP 47 ABNF: language = 2*3ALPHA / 4ALPHA / 5*8ALPHA)
            "abcd",
            "Scen",
            # Grandfathered irregular tags (RFC 5646 §2.1) -- spec §5.1
            # says loader MUST accept any valid BCP 47 tag.
            "i-klingon",
            "I-KLINGON",  # case-insensitive
            "i-ami",
            "i-default",
            "en-GB-oed",
            "sgn-BE-FR",
            # Grandfathered regular tags
            "art-lojban",
            "zh-guoyu",
            "zh-min-nan",
            "no-bok",
        ],
    )
    def test_accepts_well_formed(self, tag: str) -> None:
        assert is_valid_bcp47(tag)

    # --------------- empty / type ---------------
    def test_rejects_empty(self) -> None:
        assert not is_valid_bcp47("")

    def test_rejects_none(self) -> None:
        assert not is_valid_bcp47(None)  # type: ignore[arg-type]

    # --------------- malformed subtags ---------------
    def test_rejects_empty_subtag(self) -> None:
        assert not is_valid_bcp47("en--US")
        assert not is_valid_bcp47("en-")
        assert not is_valid_bcp47("-en")

    def test_rejects_illegal_chars(self) -> None:
        assert not is_valid_bcp47("en_US")   # underscore
        assert not is_valid_bcp47("en-U$")   # symbol
        assert not is_valid_bcp47("en-üü")  # non-ASCII

    def test_rejects_oversized_subtag(self) -> None:
        assert not is_valid_bcp47("abcdefghi")  # 9 chars

    def test_rejects_single_char_non_x(self) -> None:
        # Single-char first subtag that is not 'x' (privateuse) and is not
        # part of a grandfathered tag.
        assert not is_valid_bcp47("a")

    def test_rejects_empty_privateuse_block(self) -> None:
        assert not is_valid_bcp47("en-x")
        assert not is_valid_bcp47("x")

    def test_rejects_empty_extension_block(self) -> None:
        # A singleton must be followed by at least one 2-8 subtag.
        assert not is_valid_bcp47("en-u")
