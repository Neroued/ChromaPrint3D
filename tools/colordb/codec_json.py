"""
JSON encoding / decoding per spec §10.3.

This module deals with **bytes <-> dict tree** only. Schema validation
and dataclass construction live in :mod:`tools.colordb.validator` and
:mod:`tools.colordb.codec` respectively; keeping codec modules free of
semantic checks allows the codec layer to be swapped without affecting
validation.
"""

from __future__ import annotations

import json
from typing import Any, Dict

__all__ = ["decode_json", "encode_json"]


_UTF8_BOM = b"\xef\xbb\xbf"


def decode_json(buf: bytes) -> Dict[str, Any]:
    """Parse ``buf`` as UTF-8 JSON, tolerating an optional leading BOM.

    Returns a plain ``dict`` tree suitable for the validator. The top
    level is **not** required to be a dict at this layer; the validator
    produces a user-facing error for non-object documents.
    """
    if buf.startswith(_UTF8_BOM):
        buf = buf[len(_UTF8_BOM):]
    return json.loads(buf.decode("utf-8"))


def encode_json(
    doc: Any,
    *,
    indent: int = 2,
    write_bom: bool = False,
) -> bytes:
    """Serialize ``doc`` to UTF-8 JSON bytes.

    ``ensure_ascii=False`` to preserve Unicode in display names;
    ``sort_keys=False`` to keep palette / entry order stable (the spec
    requires palette index stability because recipes reference indices).
    """
    text = json.dumps(
        doc,
        ensure_ascii=False,
        indent=indent,
        sort_keys=False,
        allow_nan=False,
    )
    if not text.endswith("\n"):
        text += "\n"
    body = text.encode("utf-8")
    return (_UTF8_BOM + body) if write_bom else body
