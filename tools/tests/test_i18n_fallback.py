"""Tests for spec §5.2 / §5.8 fallback selection."""

from __future__ import annotations

import pytest

from tools.colordb.locale_tag import fallback, fallback_chain, spec_canonicalize


class TestFallbackChain:
    def test_bare_language(self) -> None:
        assert fallback_chain("en") == ["en"]

    def test_language_region(self) -> None:
        assert fallback_chain("en-US") == ["en-US", "en"]

    def test_language_script(self) -> None:
        assert fallback_chain("zh-Hant") == ["zh-Hant", "zh"]

    def test_language_script_region(self) -> None:
        assert fallback_chain("zh-Hant-TW") == ["zh-Hant-TW", "zh-Hant", "zh"]

    def test_mixed_case_input_canonicalized(self) -> None:
        assert fallback_chain("EN-gb") == ["en-GB", "en"]

    def test_extension_included_then_trimmed(self) -> None:
        # Extensions lowercase and still participate in right-to-left trim.
        assert fallback_chain("EN-US-U-CA-GREGORY") == [
            "en-US-u-ca-gregory",
            "en-US-u-ca",
            "en-US-u",
            "en-US",
            "en",
        ]

    def test_privateuse_included_then_trimmed(self) -> None:
        assert fallback_chain("EN-X-FOO") == ["en-x-foo", "en-x", "en"]

    def test_empty(self) -> None:
        assert fallback_chain("") == []


class TestFallbackLookup:
    @pytest.fixture
    def entry(self) -> dict:
        # Spec §5.8 sample entry.
        return {
            "display_name": "Scarlet Red",
            "display_name_localized": {
                spec_canonicalize("en"): "Scarlet Red",
                spec_canonicalize("zh-CN"): "大红",
            },
        }

    def test_exact_zh_cn(self, entry: dict) -> None:
        assert (
            fallback(entry["display_name"], entry["display_name_localized"], "zh-CN")
            == "大红"
        )

    def test_zh_tw_falls_back_to_base(self, entry: dict) -> None:
        # zh-TW -> zh (not present) -> base "Scarlet Red"
        # MUST NOT fall through to zh-CN (different script / region branch).
        assert (
            fallback(entry["display_name"], entry["display_name_localized"], "zh-TW")
            == "Scarlet Red"
        )

    def test_zh_hant_does_not_cross_to_zh_cn(self, entry: dict) -> None:
        assert (
            fallback(entry["display_name"], entry["display_name_localized"], "zh-Hant")
            == "Scarlet Red"
        )

    def test_en_us_hits_en_primary(self, entry: dict) -> None:
        assert (
            fallback(entry["display_name"], entry["display_name_localized"], "en-US")
            == "Scarlet Red"
        )

    def test_en_gb_with_mixed_case(self, entry: dict) -> None:
        assert (
            fallback(entry["display_name"], entry["display_name_localized"], "EN-gb")
            == "Scarlet Red"
        )

    def test_unrelated_locale_falls_back(self, entry: dict) -> None:
        assert (
            fallback(entry["display_name"], entry["display_name_localized"], "de")
            == "Scarlet Red"
        )

    def test_ja_jp_falls_back_to_base(self, entry: dict) -> None:
        assert (
            fallback(entry["display_name"], entry["display_name_localized"], "ja-JP")
            == "Scarlet Red"
        )


class TestFallbackReadOnly:
    def test_fallback_does_not_mutate_map(self) -> None:
        loc = {"en": "A"}
        snapshot = dict(loc)
        fallback("base", loc, "fr-FR")
        assert loc == snapshot

    def test_no_localized_returns_base(self) -> None:
        assert fallback("base", None, "en") == "base"
        assert fallback("base", {}, "en") == "base"
