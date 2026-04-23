"""Tests for spec §10.2 encoding auto-detection (and the "no fallback" stance)."""

from __future__ import annotations

from pathlib import Path

import msgpack  # type: ignore[import-not-found]
import pytest

from tools.colordb import ColorDBValidationError, load_bytes
from tools.colordb.codec import detect_encoding, load_doc

FIXTURES = Path(__file__).parent / "fixtures"


class TestDetectEncoding:
    def test_brace_is_json(self) -> None:
        assert detect_encoding(b"{}") == "json"

    @pytest.mark.parametrize("ws", [b"\t", b"\n", b"\r", b" "])
    def test_leading_whitespace_is_json(self, ws: bytes) -> None:
        assert detect_encoding(ws + b"{}") == "json"

    def test_bom_then_json(self) -> None:
        assert detect_encoding(b"\xef\xbb\xbf{}") == "json"

    def test_bom_only(self) -> None:
        # Bare BOM has no body; we treat it as JSON so the parser emits
        # a meaningful error rather than MessagePack's opaque one.
        assert detect_encoding(b"\xef\xbb\xbf") == "json"

    @pytest.mark.parametrize(
        "byte",
        [0x80, 0x81, 0x8F, 0xDE, 0xDF],
    )
    def test_fixmap_map16_map32_is_msgpack(self, byte: int) -> None:
        # Any byte outside the JSON-start set is MessagePack.
        assert detect_encoding(bytes([byte])) == "msgpack"


class TestRealFiles:
    def test_appendix_a_json(self) -> None:
        buf = (FIXTURES / "appendix_a.json").read_bytes()
        _, enc = load_doc(buf)
        assert enc == "json"

    def test_msgpack_small_map(self) -> None:
        # Build a synthetic fixmap: {"a": 1}; first byte is 0x81.
        buf = msgpack.packb({"a": 1}, use_bin_type=True)
        assert buf[0] == 0x81
        _, enc = load_doc(buf)
        assert enc == "msgpack"


class TestNoCrossFallback:
    def test_json_parse_failure_does_not_fall_back_to_msgpack(self) -> None:
        # Leading '{' forces the JSON path. Invalid JSON body will
        # surface a parse error rather than silently try msgpack.
        buf = b"{not-json"
        with pytest.raises(ColorDBValidationError) as excinfo:
            load_bytes(buf)
        # The wrapped error must be a JSON parse failure, not msgpack.
        rules = {e.rule for e in excinfo.value.report.errors}
        assert rules == {"S-json-parse"}

    def test_msgpack_parse_failure_does_not_fall_back_to_json(self) -> None:
        # A MessagePack map-start byte forces the msgpack path. A
        # truncated / corrupt body after it must surface a msgpack
        # parse error and must NOT retry the JSON codec.
        # 0x81 = fixmap with 1 entry; 0xC1 is a reserved byte that
        # derails the msgpack decoder mid-parse.
        buf = b"\x81\xc1\xff\xff"
        with pytest.raises(ColorDBValidationError) as excinfo:
            load_bytes(buf)
        rules = {e.rule for e in excinfo.value.report.errors}
        # Must be a msgpack parse failure, not a JSON parse failure,
        # not an unrecognized-encoding verdict.
        assert rules == {"S-msgpack-parse"}


class TestUnrecognizedEncoding:
    """Spec §10.2: a buffer whose first byte is neither JSON-start
    nor MessagePack map family MUST be rejected up front with
    ``S-unrecognized-encoding`` and MUST NOT be fed to either codec."""

    def _expect_unrecognized(self, buf: bytes) -> None:
        with pytest.raises(ColorDBValidationError) as excinfo:
            load_bytes(buf)
        rules = {e.rule for e in excinfo.value.report.errors}
        assert rules == {"S-unrecognized-encoding"}, (buf, rules)

    def test_empty_buffer_rejected(self) -> None:
        self._expect_unrecognized(b"")

    def test_null_byte_rejected(self) -> None:
        # 0x00 is positive fixint 0 -- not a valid top-level for a
        # ColorDB map, and not in any accepted start-byte set.
        self._expect_unrecognized(b"\x00")

    def test_reserved_c1_byte_rejected(self) -> None:
        # Historical regression: 0xC1 (reserved) used to be sent to
        # the msgpack decoder; the strict gate now rejects it up front.
        self._expect_unrecognized(b"\xc1\xff\xff")

    def test_yaml_like_document_rejected(self) -> None:
        # First byte 0x2D ('-') is neither JSON-start nor msgpack map.
        self._expect_unrecognized(b"---\nname: foo\n")

    def test_binary_header_rejected(self) -> None:
        # 0xFF is not in any accepted start-byte set.
        self._expect_unrecognized(b"\xff\xfe\x00")

    def test_bom_then_null_byte_rejected(self) -> None:
        # BOM must be stripped first, then the post-BOM byte is checked.
        self._expect_unrecognized(b"\xef\xbb\xbf\x00")
