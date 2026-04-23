"""
High-level load / dump entry points per spec §10.2 (encoding
auto-detection) and the dataclass construction contract.

``load_bytes`` / ``load_path`` produce validated :class:`ColorDB`
instances. ``dump_bytes`` / ``dump_path`` serialize a :class:`ColorDB`
back to the chosen encoding. Intermediate dict-level helpers
``load_doc`` / ``dump_doc`` are exposed for tests that want to inspect
the raw dict tree without going through validation.

Auto-detection (§10.2): after skipping a possible UTF-8 BOM, the first
byte determines the encoding.

- ``{`` (0x7B) or ASCII whitespace (``\\t \\n \\r SP``) -> JSON
- MessagePack map family (``0x80`` - ``0x8F`` / ``0xDE`` / ``0xDF``) -> MessagePack
- otherwise -> **unrecognized encoding** (MUST reject per spec §10.2)

``detect_encoding`` continues to return ``"json"`` or ``"msgpack"``
based on whether the first byte is in the JSON-start set (for backward
compatibility with tests that probe raw bytes). ``load_doc`` adds a
stricter gate: if the first byte is neither JSON-start nor MessagePack
map-start, it raises a :class:`ColorDBValidationError` carrying the
``S-unrecognized-encoding`` issue *without* attempting either codec.

**No fallback**: once a format is chosen, the loader will not retry
the other codec if parsing fails. This matches the spec's "content
determines format" language.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from .codec_json import decode_json, encode_json
from .codec_msgpack import (
    decode_msgpack,
    encode_msgpack,
    floats_to_lab_bin,
    lab_bin_to_floats,
)
from .locale_tag import spec_canonicalize
from .model import ColorDB, Defaults, Entry, PaletteEntry, Section
from .validator import (
    ColorDBValidationError,
    ValidationIssue,
    ValidationReport,
    validate,
)

__all__ = [
    "Encoding",
    "detect_encoding",
    "load_bytes",
    "load_path",
    "load_doc",
    "dump_bytes",
    "dump_path",
    "dump_doc",
]


Encoding = str  # "json" | "msgpack"

_UTF8_BOM = b"\xef\xbb\xbf"
_JSON_START_BYTES = frozenset({0x7B, 0x09, 0x0A, 0x0D, 0x20})
# MessagePack map family per spec §10.2: fixmap (0x80-0x8F), map 16 (0xDE),
# map 32 (0xDF). A spec-conforming ColorDB msgpack document MUST start with
# one of these because the top level is always a map.
_MSGPACK_MAP_START_BYTES = frozenset(
    set(range(0x80, 0x90)) | {0xDE, 0xDF}
)


def detect_encoding(buf: bytes) -> Encoding:
    """Return ``"json"`` or ``"msgpack"`` per spec §10.2.

    Kept for tests and callers that want a best-effort encoding guess
    from raw bytes. Does not validate that the chosen encoding will
    actually parse; use :func:`load_doc` for the spec-strict path that
    rejects unrecognized byte streams up front.
    """
    if not buf:
        raise ValueError("cannot detect encoding of empty buffer")
    probe = buf
    if probe.startswith(_UTF8_BOM):
        probe = probe[len(_UTF8_BOM):]
        if not probe:
            # A bare BOM is not a valid JSON document; treat as JSON so
            # that the downstream parser emits a meaningful error
            # instead of MessagePack's opaque one.
            return "json"
    first = probe[0]
    return "json" if first in _JSON_START_BYTES else "msgpack"


# --------------------------------------------------------------------------
# Load.
# --------------------------------------------------------------------------


def load_doc(buf: bytes) -> Tuple[Dict[str, Any], Encoding]:
    """Decode ``buf`` into a raw dict tree + encoding tag.

    Does **not** validate schema semantics; it does, however, surface
    encoding-level parse failures as :class:`ColorDBValidationError`
    so that callers receive a single, unified exception type. This is
    consistent with the spec §10.2 "no cross-encoding fallback"
    stance: once the encoding is picked by the first byte, a parse
    failure on that codec is terminal and must not retry the other
    codec.

    Spec §10.2 strict encoding gate: the first byte (after an optional
    UTF-8 BOM) MUST be either a JSON-start byte or a MessagePack map-family
    byte. Any other leading byte is rejected with ``S-unrecognized-encoding``
    without attempting to parse via either codec.
    """
    if not buf:
        issue = ValidationIssue(
            rule="S-unrecognized-encoding",
            severity="MUST",
            path="$",
            message="empty buffer is neither JSON nor MessagePack",
        )
        raise ColorDBValidationError(
            ValidationReport(errors=(issue,), warnings=())
        )

    probe = buf[len(_UTF8_BOM):] if buf.startswith(_UTF8_BOM) else buf
    if not probe:
        # Bare BOM: keep the pre-existing behaviour of falling into the
        # JSON parser so the user gets a meaningful "empty document"
        # error from the JSON codec. (This matches the legacy
        # ``test_bom_only`` contract.)
        enc: Encoding = "json"
    else:
        first = probe[0]
        if first in _JSON_START_BYTES:
            enc = "json"
        elif first in _MSGPACK_MAP_START_BYTES:
            enc = "msgpack"
        else:
            issue = ValidationIssue(
                rule="S-unrecognized-encoding",
                severity="MUST",
                path="$",
                message=(
                    f"first byte 0x{first:02X} is neither JSON-start "
                    f"(0x7B / whitespace) nor MessagePack map family "
                    f"(0x80-0x8F / 0xDE / 0xDF)"
                ),
            )
            raise ColorDBValidationError(
                ValidationReport(errors=(issue,), warnings=())
            )

    try:
        if enc == "json":
            doc = decode_json(buf)
        else:
            doc = decode_msgpack(buf)
    except Exception as e:  # noqa: BLE001 - codec failures are surfaced uniformly
        rule = "S-json-parse" if enc == "json" else "S-msgpack-parse"
        issue = ValidationIssue(
            rule=rule,
            severity="MUST",
            path="$",
            message=f"{enc} decode failed: {type(e).__name__}: {e}",
        )
        raise ColorDBValidationError(
            ValidationReport(errors=(issue,), warnings=())
        ) from e
    return doc, enc


def load_bytes(buf: bytes) -> ColorDB:
    """Load, validate and construct a :class:`ColorDB` from ``buf``.

    Raises :class:`ColorDBValidationError` on any MUST violation.
    """
    doc, enc = load_doc(buf)
    report = validate(doc)
    report.raise_if_errors()
    return _build_color_db(doc, enc)


def load_path(path: Union[str, "Path"]) -> ColorDB:
    """Read ``path`` and call :func:`load_bytes`."""
    return load_bytes(Path(path).read_bytes())


# --------------------------------------------------------------------------
# Dump.
# --------------------------------------------------------------------------


def dump_doc(
    db: ColorDB,
    *,
    recipe_as_array: bool = False,
    lab_as_bin: bool = False,
    lab_decimals: Optional[int] = None,
) -> Dict[str, Any]:
    """Serialize ``db`` to a plain dict tree (for codec layer).

    Flags control the MessagePack-specific field shapes. They are
    ignored by the JSON encoder because JSON cannot represent bytes
    natively (V20 forbids bin ``lab`` in JSON anyway).
    """
    out: Dict[str, Any] = {
        "schema_version": db.schema_version,
        "name": db.name,
    }
    if db.vendor is not None:
        out["vendor"] = db.vendor
    if db.material_type is not None:
        out["material_type"] = db.material_type
    out["palette"] = [_palette_entry_to_dict(e) for e in db.palette]
    out["defaults"] = _defaults_to_dict(db.defaults)
    if db.meta:
        out["meta"] = dict(db.meta)
    out["sections"] = [
        _section_to_dict(s, recipe_as_array=recipe_as_array,
                         lab_as_bin=lab_as_bin, lab_decimals=lab_decimals)
        for s in db.sections
    ]
    return out


def dump_bytes(
    db: ColorDB,
    *,
    encoding: Encoding = "json",
    recipe_as_array: bool = False,
    lab_as_bin: bool = False,
    lab_decimals: Optional[int] = None,
    indent: int = 2,
    write_bom: bool = False,
) -> bytes:
    """Serialize ``db`` to bytes in the requested encoding."""
    if encoding == "json":
        # JSON cannot carry bin lab/recipe; force array form regardless
        # of the flags to satisfy §10.3 / V20.
        doc = dump_doc(db, recipe_as_array=True, lab_as_bin=False,
                       lab_decimals=lab_decimals)
        return encode_json(doc, indent=indent, write_bom=write_bom)
    if encoding == "msgpack":
        doc = dump_doc(db, recipe_as_array=recipe_as_array,
                       lab_as_bin=lab_as_bin, lab_decimals=lab_decimals)
        return encode_msgpack(doc)
    raise ValueError(f"unsupported encoding: {encoding!r}")


def dump_path(
    db: ColorDB,
    path: Union[str, "Path"],
    *,
    encoding: Optional[Encoding] = None,
    **kwargs: Any,
) -> None:
    """Write ``db`` to ``path``.

    If ``encoding`` is omitted, it is inferred from the file suffix:
    ``.json`` -> JSON, anything else -> MessagePack. The spec does not
    require this behaviour; it is a convenience for the reference
    implementation.
    """
    p = Path(path)
    if encoding is None:
        encoding = "json" if p.suffix.lower() == ".json" else "msgpack"
    p.write_bytes(dump_bytes(db, encoding=encoding, **kwargs))


# --------------------------------------------------------------------------
# Internal: dict tree -> ColorDB dataclasses.
# --------------------------------------------------------------------------


def _build_color_db(doc: Mapping[str, Any], enc: Encoding) -> ColorDB:
    palette = tuple(_build_palette_entry(e) for e in doc["palette"])
    defaults = _build_defaults(doc["defaults"])
    sections = tuple(
        _build_section(s, enc) for s in doc["sections"]
    )
    return ColorDB(
        schema_version=int(doc["schema_version"]),
        name=str(doc["name"]),
        palette=palette,
        defaults=defaults,
        sections=sections,
        vendor=doc["vendor"] if isinstance(doc.get("vendor"), str) else None,
        material_type=(
            doc["material_type"]
            if isinstance(doc.get("material_type"), str)
            else None
        ),
        meta=dict(doc["meta"]) if isinstance(doc.get("meta"), Mapping) else {},
    )


def _build_palette_entry(raw: Mapping[str, Any]) -> PaletteEntry:
    localized_raw = raw.get("display_name_localized") or {}
    # Keys are canonicalized so that fallback lookups are consistent
    # regardless of the source file's key casing (spec §5.1.1).
    localized = {
        spec_canonicalize(str(k)): str(v) for k, v in localized_raw.items()
    }
    return PaletteEntry(
        channel_class=str(raw["channel_class"]),
        display_name=str(raw["display_name"]),
        material=str(raw["material"]),
        hex_color=str(raw["hex_color"]) if "hex_color" in raw else None,
        display_name_localized=localized,
    )


def _build_defaults(raw: Mapping[str, Any]) -> Defaults:
    return Defaults(
        color_layers=int(raw["color_layers"]),
        layer_height_mm=float(raw["layer_height_mm"]),
        line_width_mm=float(raw["line_width_mm"]),
        base_layers=int(raw["base_layers"]),
        base_channel_idx=int(raw["base_channel_idx"]),
    )


def _build_section(raw: Mapping[str, Any], enc: Encoding) -> Section:
    entries = tuple(_build_entry(e, enc) for e in raw["entries"])
    return Section(
        type=str(raw["type"]),
        entries=entries,
        color_layers=(
            int(raw["color_layers"]) if "color_layers" in raw else None
        ),
        layer_height_mm=(
            float(raw["layer_height_mm"])
            if "layer_height_mm" in raw
            else None
        ),
        line_width_mm=(
            float(raw["line_width_mm"]) if "line_width_mm" in raw else None
        ),
        base_layers=(
            int(raw["base_layers"]) if "base_layers" in raw else None
        ),
        base_channel_idx=(
            int(raw["base_channel_idx"])
            if "base_channel_idx" in raw
            else None
        ),
        threshold=(
            float(raw["threshold"]) if "threshold" in raw else None
        ),
        margin=(float(raw["margin"]) if "margin" in raw else None),
    )


def _build_entry(raw: Mapping[str, Any], enc: Encoding) -> Entry:
    lab_raw = raw["lab"]
    if isinstance(lab_raw, (bytes, bytearray, memoryview)):
        lab = lab_bin_to_floats(bytes(lab_raw))
    else:
        lab = tuple(float(v) for v in lab_raw)  # type: ignore[assignment]
    recipe_raw = raw["recipe"]
    if isinstance(recipe_raw, (bytes, bytearray, memoryview)):
        recipe = tuple(int(v) for v in bytes(recipe_raw))
    else:
        recipe = tuple(int(v) for v in recipe_raw)
    return Entry(lab=lab, recipe=recipe)


# --------------------------------------------------------------------------
# Internal: ColorDB dataclasses -> dict tree.
# --------------------------------------------------------------------------


def _palette_entry_to_dict(entry: PaletteEntry) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "channel_class": entry.channel_class,
        "display_name": entry.display_name,
        "material": entry.material,
    }
    if entry.hex_color is not None:
        out["hex_color"] = entry.hex_color
    if entry.display_name_localized:
        out["display_name_localized"] = dict(entry.display_name_localized)
    return out


def _defaults_to_dict(defaults: Defaults) -> Dict[str, Any]:
    return {
        "color_layers": defaults.color_layers,
        "layer_height_mm": defaults.layer_height_mm,
        "line_width_mm": defaults.line_width_mm,
        "base_layers": defaults.base_layers,
        "base_channel_idx": defaults.base_channel_idx,
    }


def _section_to_dict(
    section: Section,
    *,
    recipe_as_array: bool,
    lab_as_bin: bool,
    lab_decimals: Optional[int],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"type": section.type}
    for name in (
        "color_layers",
        "layer_height_mm",
        "line_width_mm",
        "base_layers",
        "base_channel_idx",
    ):
        v = getattr(section, name)
        if v is not None:
            out[name] = v
    if section.threshold is not None:
        out["threshold"] = section.threshold
    if section.margin is not None:
        out["margin"] = section.margin
    out["entries"] = [
        _entry_to_dict(
            e,
            recipe_as_array=recipe_as_array,
            lab_as_bin=lab_as_bin,
            lab_decimals=lab_decimals,
        )
        for e in section.entries
    ]
    return out


def _entry_to_dict(
    entry: Entry,
    *,
    recipe_as_array: bool,
    lab_as_bin: bool,
    lab_decimals: Optional[int],
) -> Dict[str, Any]:
    if lab_as_bin:
        lab_out: Any = floats_to_lab_bin(entry.lab)
    elif lab_decimals is not None:
        lab_out = [round(v, lab_decimals) for v in entry.lab]
    else:
        lab_out = list(entry.lab)
    if recipe_as_array:
        recipe_out: Any = list(entry.recipe)
    else:
        # Default bytes form (valid in MessagePack only; JSON forces
        # array via dump_bytes()).
        recipe_out = bytes(entry.recipe)
    return {"lab": lab_out, "recipe": recipe_out}
