"""Tests for tools.colordb.codec_json (spec §10.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.colordb import load_bytes, load_doc
from tools.colordb.codec_json import decode_json, encode_json

FIXTURES = Path(__file__).parent / "fixtures"
_BOM = b"\xef\xbb\xbf"


class TestDecodeJson:
    def test_plain_utf8(self) -> None:
        data = {"schema_version": 1, "name": "x"}
        buf = json.dumps(data).encode("utf-8")
        assert decode_json(buf) == data

    def test_accepts_leading_bom(self) -> None:
        data = {"schema_version": 1}
        buf = _BOM + json.dumps(data).encode("utf-8")
        assert decode_json(buf) == data


class TestEncodeJson:
    def test_ensure_ascii_false_preserves_unicode(self) -> None:
        out = encode_json({"name": "大红"})
        # Must contain the raw UTF-8 bytes, not \u escapes.
        assert "大红".encode("utf-8") in out
        assert b"\\u" not in out

    def test_top_level_is_object(self) -> None:
        out = encode_json({"a": 1})
        text = out.decode("utf-8")
        assert text.lstrip().startswith("{")

    def test_indent_2(self) -> None:
        out = encode_json({"a": 1, "b": [1, 2]})
        assert b"  " in out

    def test_no_sort_keys(self) -> None:
        # palette / entry order must be preserved across encoding.
        out = encode_json({"zeta": 1, "alpha": 2})
        text = out.decode("utf-8")
        assert text.index("zeta") < text.index("alpha")

    def test_write_bom(self) -> None:
        out = encode_json({"a": 1}, write_bom=True)
        assert out.startswith(_BOM)

    def test_no_bom_default(self) -> None:
        out = encode_json({"a": 1})
        assert not out.startswith(_BOM)


class TestAppendixAJsonCodec:
    def test_load_doc_preserves_structure(self) -> None:
        buf = (FIXTURES / "appendix_a.json").read_bytes()
        doc, enc = load_doc(buf)
        assert enc == "json"
        assert doc["schema_version"] == 1
        assert doc["palette"][0]["channel_class"] == "White"

    def test_lab_precision_preserved(self) -> None:
        # Spec §10.3 SHOULD "keep at least 2 decimals"; our encoder
        # preserves the original parse precision by default.
        buf = (FIXTURES / "appendix_a.json").read_bytes()
        db = load_bytes(buf)
        first_lab = db.sections[0].entries[0].lab
        assert first_lab[0] == pytest.approx(93.87)
        assert first_lab[1] == pytest.approx(-1.51)
        assert first_lab[2] == pytest.approx(1.32)
