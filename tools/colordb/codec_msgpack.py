"""
MessagePack encoding / decoding per spec §10.4.

Unpack options:

- ``raw=False`` so that ``str`` family unpacks to Python ``str`` (spec
  §10.4 requires str-family strings).
- ``strict_map_key=False`` to tolerate non-string map keys if present
  in third-party dumps (we still reject them at validation time).

Pack options:

- ``use_bin_type=True`` to write bytes as bin type and ``str`` as str
  family (required by spec §10.4).

This module does **not** massage bin-encoded ``lab`` or ``recipe``
fields into lists; it returns the raw dict tree so that the validator
can inspect both representations and enforce V15 / V20 / V25. Higher
layers (`tools.colordb.codec`) convert bin forms to canonical
dataclass values after validation succeeds.
"""

from __future__ import annotations

import struct
from typing import Any, Dict

import msgpack  # type: ignore[import-not-found]

__all__ = [
    "decode_msgpack",
    "encode_msgpack",
    "lab_bin_to_floats",
    "floats_to_lab_bin",
]


def decode_msgpack(buf: bytes) -> Dict[str, Any]:
    """Parse ``buf`` as MessagePack, returning the raw dict tree."""
    return msgpack.unpackb(
        buf,
        raw=False,
        strict_map_key=False,
        use_list=True,
    )


def encode_msgpack(
    doc: Any,
) -> bytes:
    """Serialize ``doc`` to MessagePack bytes with spec-compliant flags.

    Callers that want ``lab`` as a 12-byte bin or ``recipe`` as an
    array must convert the relevant fields before passing them in; the
    :mod:`tools.colordb.codec` layer provides helpers for that.
    """
    return msgpack.packb(
        doc,
        use_bin_type=True,
    )


# --------------------------------------------------------------------------
# Bin helpers for lab and recipe encoding conversions.
# --------------------------------------------------------------------------

_LAB_STRUCT = struct.Struct("<fff")


def lab_bin_to_floats(buf: bytes) -> "tuple[float, float, float]":
    """Decode a 12-byte lab bin (3 x float32 little-endian) to 3 floats.

    Raises :class:`ValueError` if the buffer is not exactly 12 bytes
    long; callers (i.e. the validator) should surface this as V25.
    """
    if len(buf) != 12:
        raise ValueError(
            f"lab bin must be exactly 12 bytes (got {len(buf)})"
        )
    return tuple(float(v) for v in _LAB_STRUCT.unpack(buf))  # type: ignore[return-value]


def floats_to_lab_bin(values: "tuple[float, float, float]") -> bytes:
    """Encode 3 floats as a 12-byte lab bin (3 x float32 LE)."""
    if len(values) != 3:
        raise ValueError("lab must be 3 floats")
    return _LAB_STRUCT.pack(*values)
