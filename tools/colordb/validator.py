"""
Validation per spec §11 (V1-V25) plus structural checks implied by §2 /
§3.1 / §6 / §8.1.

The validator runs against the **raw parsed document** (a plain
``dict`` / ``list`` tree from the codec layer) rather than against
constructed dataclasses. This lets it distinguish "field absent",
"field wrong type", "field wrong value" with full precision and keeps
V13's "silently ignore unknown fields" logic at the model layer where
it belongs.

Severity handling:

- ``"MUST"`` errors make the document invalid (CLI exit 1).
- ``"SHOULD"`` warnings do not invalidate the document (CLI exit 2,
  or exit 1 under ``--strict``).

This module also exports :func:`match_preset` for §3.5 multiset
preset matching. It is **not** a validation rule (§11 does not include
preset matching) but lives here because it shares the palette shape.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from .constants import (
    AIR_INDEX,
    CHANNEL_CLASSES_SET,
    DEFAULT_IGNORABLE_CODE_POINTS,
    HEX_COLOR_PATTERN,
    MAX_PALETTE_SIZE,
    SCHEMA_VERSION,
    SECTION_TYPES,
    STANDARD_PRESETS,
)
from .locale_tag import is_valid_bcp47, spec_canonicalize
from .normalize import normalize

__all__ = [
    "ColorDBValidationError",
    "ValidationIssue",
    "ValidationReport",
    "validate",
    "match_preset",
]


_HEX_COLOR_RE = re.compile(HEX_COLOR_PATTERN)


# --------------------------------------------------------------------------
# Issue / report types.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ValidationIssue:
    """A single rule violation.

    ``rule`` is either a spec rule identifier (``"V1"`` - ``"V25"``)
    or a structural tag (``"S-name"``, ``"S-hex_color"`` ...) for
    checks implied by the spec's field tables but not listed in
    §11. ``severity`` is ``"MUST"`` or ``"SHOULD"``; ``path`` is a
    human-readable JSONPath-like locator.
    """

    rule: str
    severity: str
    path: str
    message: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "rule": self.rule,
            "severity": self.severity,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True)
class ValidationReport:
    """Result of a :func:`validate` call."""

    errors: Tuple[ValidationIssue, ...]
    warnings: Tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise ColorDBValidationError(self)


class ColorDBValidationError(Exception):
    """Raised by :meth:`ValidationReport.raise_if_errors`.

    Carries the whole report so that callers (loaders, CLI) can render
    all issues rather than only the first.
    """

    def __init__(self, report: "ValidationReport") -> None:
        self.report = report
        summary = "; ".join(
            f"{e.rule} @ {e.path}: {e.message}" for e in report.errors
        )
        super().__init__(summary or "ColorDB validation failed")


# --------------------------------------------------------------------------
# Internal collector.
# --------------------------------------------------------------------------


class _Collector:
    def __init__(self) -> None:
        self.errors: List[ValidationIssue] = []
        self.warnings: List[ValidationIssue] = []

    def must(self, rule: str, path: str, message: str) -> None:
        self.errors.append(
            ValidationIssue(rule=rule, severity="MUST", path=path, message=message)
        )

    def should(self, rule: str, path: str, message: str) -> None:
        self.warnings.append(
            ValidationIssue(rule=rule, severity="SHOULD", path=path, message=message)
        )


# --------------------------------------------------------------------------
# Type predicates (JSON / msgpack bridging).
# --------------------------------------------------------------------------


def _is_bool(x: Any) -> bool:
    return isinstance(x, bool)


def _is_int(x: Any) -> bool:
    # Reject bool (which is a subclass of int) so that ``True``/``False``
    # do not silently satisfy int fields.
    return isinstance(x, int) and not _is_bool(x)


def _is_number(x: Any) -> bool:
    """Return True for JSON int or float (matches the spec §6 / §8.1
    "float" fields per plan: accept int literals and normalize to
    float at the model layer)."""
    return (_is_int(x) or isinstance(x, float)) and not _is_bool(x)


def _is_string(x: Any) -> bool:
    return isinstance(x, str)


def _is_mapping(x: Any) -> bool:
    return isinstance(x, Mapping)


def _is_sequence(x: Any) -> bool:
    return isinstance(x, (list, tuple)) and not isinstance(x, (str, bytes))


# --------------------------------------------------------------------------
# Public API.
# --------------------------------------------------------------------------


def validate(doc: Any) -> ValidationReport:
    """Validate a parsed ColorDB document against the spec.

    The input is expected to be a plain ``dict`` tree (as produced by
    ``json.loads`` or ``msgpack.unpackb(raw=False)``), not a
    dataclass instance.
    """
    c = _Collector()

    if not _is_mapping(doc):
        c.must("S-toplevel", "$", "top-level must be object / map")
        return ValidationReport(tuple(c.errors), tuple(c.warnings))

    _validate_top_level(doc, c)
    palette_len = _palette_length(doc)
    defaults = doc.get("defaults") if _is_mapping(doc.get("defaults")) else None

    _validate_palette(doc, c)
    _validate_defaults(doc, c, palette_len)

    resolved_defaults = _resolved_defaults(defaults)
    _validate_sections(doc, c, palette_len, resolved_defaults)

    return ValidationReport(tuple(c.errors), tuple(c.warnings))


# --------------------------------------------------------------------------
# Top-level structure (§2).
# --------------------------------------------------------------------------


def _validate_top_level(doc: Mapping[str, Any], c: _Collector) -> None:
    # V1: schema_version == 1 (integer)
    sv = doc.get("schema_version")
    if "schema_version" not in doc:
        c.must("V1", "$.schema_version", "schema_version is required")
    elif not _is_int(sv) or sv != SCHEMA_VERSION:
        c.must("V1", "$.schema_version", f"schema_version must be integer {SCHEMA_VERSION}")

    # name (string, required; spec §2 does not require non-empty)
    name = doc.get("name")
    if "name" not in doc:
        c.must("S-name", "$.name", "name is required")
    elif not _is_string(name):
        c.must("S-name", "$.name", "name must be string")

    # vendor / material_type (string, optional)
    if "vendor" in doc and not _is_string(doc["vendor"]):
        c.must("S-vendor", "$.vendor", "vendor must be string")
    if "material_type" in doc and not _is_string(doc["material_type"]):
        c.must(
            "S-material_type",
            "$.material_type",
            "material_type must be string",
        )

    # palette / defaults / sections presence & type (fields checked further down)
    if "palette" not in doc:
        c.must("S-palette-required", "$.palette", "palette is required")
    elif not _is_sequence(doc["palette"]):
        c.must("S-palette-type", "$.palette", "palette must be array")

    if "defaults" not in doc:
        c.must("S-defaults-required", "$.defaults", "defaults is required")
    elif not _is_mapping(doc["defaults"]):
        c.must("S-defaults-type", "$.defaults", "defaults must be object / map")

    if "sections" not in doc:
        c.must("S-sections-required", "$.sections", "sections is required")
    elif not _is_sequence(doc["sections"]):
        c.must("S-sections-type", "$.sections", "sections must be array")

    # meta (free-form object; spec §7)
    if "meta" in doc and not _is_mapping(doc["meta"]):
        c.must("S-meta", "$.meta", "meta must be object / map when present")


def _palette_length(doc: Mapping[str, Any]) -> int:
    p = doc.get("palette")
    if _is_sequence(p):
        return len(p)
    return 0


# --------------------------------------------------------------------------
# Palette (§3).
# --------------------------------------------------------------------------


def _validate_palette(doc: Mapping[str, Any], c: _Collector) -> None:
    palette = doc.get("palette")
    if not _is_sequence(palette):
        return  # already flagged in top-level check

    n = len(palette)
    # V2: non-empty and length <= MAX_PALETTE_SIZE
    if n == 0:
        c.must("V2", "$.palette", "palette must contain at least 1 entry")
    if n > MAX_PALETTE_SIZE:
        c.must(
            "V2",
            "$.palette",
            f"palette must contain at most {MAX_PALETTE_SIZE} entries (got {n})",
        )

    seen_keys: Dict[Tuple[str, str], int] = {}
    channel_class_counts: Dict[str, int] = {}

    for idx, entry in enumerate(palette):
        path = f"$.palette[{idx}]"
        if not _is_mapping(entry):
            c.must("S-palette-entry", path, "palette entry must be object / map")
            continue

        cc = entry.get("channel_class")
        # V16: channel_class required and in enum (case-sensitive)
        if "channel_class" not in entry:
            c.must("V16", f"{path}.channel_class", "channel_class is required")
        elif not _is_string(cc) or cc not in CHANNEL_CLASSES_SET:
            c.must(
                "V16",
                f"{path}.channel_class",
                f"channel_class must be one of the 52 enum values (case-sensitive), got {cc!r}",
            )
        else:
            channel_class_counts[cc] = channel_class_counts.get(cc, 0) + 1

        # V17: display_name required and non-empty; after §4.1 normalize
        # MUST retain at least one non-White_Space, non-Default_Ignorable
        # character. Literal and normalized checks are both MUST-level
        # under the strengthened V17.
        dn = entry.get("display_name")
        if "display_name" not in entry:
            c.must("V17", f"{path}.display_name", "display_name is required")
        elif not _is_string(dn) or not dn:
            c.must(
                "V17",
                f"{path}.display_name",
                "display_name must be a non-empty string",
            )
        else:
            _validate_display_name_content(dn, f"{path}.display_name", c)

        # material required, spec §3.1 does not demand non-empty
        mat = entry.get("material")
        if "material" not in entry:
            c.must("S-material", f"{path}.material", "material is required")
        elif not _is_string(mat):
            c.must("S-material", f"{path}.material", "material must be string")

        # hex_color optional but if present must match #RRGGBB
        if "hex_color" in entry:
            hx = entry["hex_color"]
            if not _is_string(hx) or not _HEX_COLOR_RE.match(hx):
                c.must(
                    "S-hex_color",
                    f"{path}.hex_color",
                    "hex_color must match '#RRGGBB'",
                )

        # V18 / V19: display_name_localized
        if "display_name_localized" in entry:
            _validate_localized(entry["display_name_localized"], path, c)

        # V3 key: (normalize(display_name), normalize(material))
        if _is_string(dn) and dn and _is_string(mat):
            key = (normalize(dn), normalize(mat))
            if key in seen_keys:
                other = seen_keys[key]
                c.must(
                    "V3",
                    path,
                    f"palette uniqueness key conflicts with entry[{other}]",
                )
            else:
                seen_keys[key] = idx

    # §3.2 SHOULD: channel_class should not repeat.
    for cc, count in channel_class_counts.items():
        if count > 1:
            c.should(
                "S-channel_class-unique",
                "$.palette",
                f"channel_class {cc!r} appears {count} times; consider a more specific class",
            )


def _validate_display_name_content(
    dn: str,
    path: str,
    c: _Collector,
) -> None:
    """V17 (strengthened): display_name normalized per §4.1 MUST retain
    at least one codepoint that is neither White_Space (already stripped
    by :func:`normalize`) nor Default_Ignorable_Code_Point (UCD 15.1).

    This prevents palette entries that look non-empty in source but render
    as a blank label in UI (e.g. ``\"   \"``, ``\"\u200B\u200B\"``,
    ``\"\u00AD\"``).
    """
    normalized = normalize(dn)
    if not normalized:
        c.must(
            "V17",
            path,
            "display_name is whitespace-only (empty after §4.1 normalize)",
        )
        return
    if all(ord(ch) in DEFAULT_IGNORABLE_CODE_POINTS for ch in normalized):
        c.must(
            "V17",
            path,
            "display_name contains only Default_Ignorable_Code_Point characters after §4.1 normalize",
        )


def _validate_localized(
    val: Any,
    entry_path: str,
    c: _Collector,
) -> None:
    path = f"{entry_path}.display_name_localized"
    # V18: must be object / map of BCP47 tag -> non-empty string
    if not _is_mapping(val):
        c.must("V18", path, "display_name_localized must be object / map")
        return
    seen_canonical: Dict[str, str] = {}
    for key, v in val.items():
        key_path = f"{path}[{key!r}]"
        if not _is_string(key) or not is_valid_bcp47(key):
            c.must(
                "V18",
                key_path,
                f"locale key {key!r} is not a valid BCP 47 tag",
            )
            continue
        if not _is_string(v) or not v:
            c.must(
                "V18",
                key_path,
                "localized value must be a non-empty string",
            )
            continue
        canonical = spec_canonicalize(key)
        if canonical in seen_canonical:
            c.must(
                "V19",
                key_path,
                f"locale key {key!r} collides with {seen_canonical[canonical]!r} after spec-canonicalize (both -> {canonical!r})",
            )
        else:
            seen_canonical[canonical] = key


# --------------------------------------------------------------------------
# Defaults (§6) + V4 / V5.
# --------------------------------------------------------------------------


_DEFAULTS_INT_FIELDS = ("color_layers", "base_layers", "base_channel_idx")
_DEFAULTS_FLOAT_FIELDS = ("layer_height_mm", "line_width_mm")


def _validate_defaults(
    doc: Mapping[str, Any],
    c: _Collector,
    palette_len: int,
) -> None:
    defaults = doc.get("defaults")
    if not _is_mapping(defaults):
        return  # already flagged

    # V4: required fields exist and have correct type.
    for fname in _DEFAULTS_INT_FIELDS:
        if fname not in defaults:
            c.must("V4", f"$.defaults.{fname}", f"{fname} is required")
        elif not _is_int(defaults[fname]):
            c.must(
                "V4",
                f"$.defaults.{fname}",
                f"{fname} must be int, got {type(defaults[fname]).__name__}",
            )
    for fname in _DEFAULTS_FLOAT_FIELDS:
        if fname not in defaults:
            c.must("V4", f"$.defaults.{fname}", f"{fname} is required")
        elif not _is_number(defaults[fname]):
            c.must(
                "V4",
                f"$.defaults.{fname}",
                f"{fname} must be number, got {type(defaults[fname]).__name__}",
            )

    # V5: base_channel_idx in [0, palette_len)
    bci = defaults.get("base_channel_idx")
    if _is_int(bci) and palette_len > 0:
        if not (0 <= bci < palette_len):
            c.must(
                "V5",
                "$.defaults.base_channel_idx",
                f"base_channel_idx {bci} out of range [0, {palette_len})",
            )

    # V23 (defaults portion): positivity checks.
    _validate_positivity(defaults, "$.defaults", c, require_present=True)


def _validate_positivity(
    obj: Mapping[str, Any],
    path: str,
    c: _Collector,
    *,
    require_present: bool,
) -> None:
    # color_layers > 0
    if "color_layers" in obj:
        v = obj["color_layers"]
        if _is_int(v) and v <= 0:
            c.must(
                "V23",
                f"{path}.color_layers",
                "color_layers must be > 0",
            )
    elif require_present:
        # already flagged by V4
        pass
    # layer_height_mm > 0
    if "layer_height_mm" in obj:
        v = obj["layer_height_mm"]
        if _is_number(v) and v <= 0:
            c.must(
                "V23",
                f"{path}.layer_height_mm",
                "layer_height_mm must be > 0",
            )
    # line_width_mm > 0
    if "line_width_mm" in obj:
        v = obj["line_width_mm"]
        if _is_number(v) and v <= 0:
            c.must(
                "V23",
                f"{path}.line_width_mm",
                "line_width_mm must be > 0",
            )
    # base_layers >= 0
    if "base_layers" in obj:
        v = obj["base_layers"]
        if _is_int(v) and v < 0:
            c.must(
                "V23",
                f"{path}.base_layers",
                "base_layers must be >= 0",
            )


def _resolved_defaults(defaults: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Extract the subset of defaults that we can inherit from.

    Returns values only when they are of the correct type; otherwise the
    key is missing and downstream checks that require the resolved value
    will skip (because the root-cause error was already reported).
    """
    out: Dict[str, Any] = {}
    if not _is_mapping(defaults):
        return out
    for k in _DEFAULTS_INT_FIELDS:
        if _is_int(defaults.get(k)):
            out[k] = defaults[k]
    for k in _DEFAULTS_FLOAT_FIELDS:
        if _is_number(defaults.get(k)):
            out[k] = float(defaults[k])
    return out


# --------------------------------------------------------------------------
# Sections (§8) + entries (§9).
# --------------------------------------------------------------------------


def _validate_sections(
    doc: Mapping[str, Any],
    c: _Collector,
    palette_len: int,
    resolved_defaults: Mapping[str, Any],
) -> None:
    sections = doc.get("sections")
    if not _is_sequence(sections):
        return

    identity_keys: Dict[Tuple, int] = {}

    for s_idx, section in enumerate(sections):
        s_path = f"$.sections[{s_idx}]"
        if not _is_mapping(section):
            c.must("S-section", s_path, "section must be object / map")
            continue

        # V6: type and entries required
        if "type" not in section:
            c.must("V6", f"{s_path}.type", "section.type is required")
        if "entries" not in section:
            c.must("V6", f"{s_path}.entries", "section.entries is required")

        stype = section.get("type")
        # V7: type in {measured, predicted}
        if "type" in section and (
            not _is_string(stype) or stype not in SECTION_TYPES
        ):
            c.must(
                "V7",
                f"{s_path}.type",
                f"section.type must be 'measured' or 'predicted', got {stype!r}",
            )

        # V22: override types must match defaults field types
        _validate_section_overrides(section, s_path, c)

        # V11: threshold / margin only in predicted
        for field_name in ("threshold", "margin"):
            if field_name in section:
                if not _is_number(section[field_name]):
                    c.must(
                        f"S-{field_name}",
                        f"{s_path}.{field_name}",
                        f"{field_name} must be number",
                    )
                if stype == "measured":
                    c.must(
                        "V11",
                        f"{s_path}.{field_name}",
                        f"{field_name} must only appear in predicted sections",
                    )

        # V23 on section overrides
        _validate_positivity(section, s_path, c, require_present=False)

        # Resolve config for V9 / V10 / V24 / V25 / identity key.
        resolved = _resolve_section_config(section, resolved_defaults)
        if resolved is not None:
            # V24: resolved base_channel_idx in [0, palette_len)
            bci = resolved.get("base_channel_idx")
            if bci is not None and palette_len > 0 and not (0 <= bci < palette_len):
                c.must(
                    "V24",
                    f"{s_path}.base_channel_idx",
                    f"resolved base_channel_idx {bci} out of range [0, {palette_len})",
                )

            # V8 identity key
            if stype in SECTION_TYPES:
                key = (
                    stype,
                    resolved.get("color_layers"),
                    resolved.get("layer_height_mm"),
                    resolved.get("line_width_mm"),
                    resolved.get("base_layers"),
                )
                if None not in key:
                    if key in identity_keys:
                        other = identity_keys[key]
                        c.should(
                            "V8",
                            s_path,
                            f"section identity key duplicates section[{other}]",
                        )
                    else:
                        identity_keys[key] = s_idx

        _validate_entries(section, s_path, c, palette_len, resolved)


def _validate_section_overrides(
    section: Mapping[str, Any],
    s_path: str,
    c: _Collector,
) -> None:
    for name in _DEFAULTS_INT_FIELDS:
        if name in section and not _is_int(section[name]):
            c.must(
                "V22",
                f"{s_path}.{name}",
                f"{name} override must be int",
            )
    for name in _DEFAULTS_FLOAT_FIELDS:
        if name in section and not _is_number(section[name]):
            c.must(
                "V22",
                f"{s_path}.{name}",
                f"{name} override must be number",
            )


def _resolve_section_config(
    section: Mapping[str, Any],
    resolved_defaults: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    out: Dict[str, Any] = {}
    for name in _DEFAULTS_INT_FIELDS:
        if name in section:
            if _is_int(section[name]):
                out[name] = section[name]
            else:
                # Type error already reported; skip this section for
                # resolved-config dependent checks.
                out[name] = None
        elif name in resolved_defaults:
            out[name] = resolved_defaults[name]
        else:
            out[name] = None
    for name in _DEFAULTS_FLOAT_FIELDS:
        if name in section:
            if _is_number(section[name]):
                out[name] = float(section[name])
            else:
                out[name] = None
        elif name in resolved_defaults:
            out[name] = resolved_defaults[name]
        else:
            out[name] = None
    return out


def _validate_entries(
    section: Mapping[str, Any],
    s_path: str,
    c: _Collector,
    palette_len: int,
    resolved: Optional[Mapping[str, Any]],
) -> None:
    entries = section.get("entries")
    if "entries" not in section:
        return  # already flagged
    # V21: entries must be array (empty is allowed).
    if not _is_sequence(entries):
        c.must("V21", f"{s_path}.entries", "entries must be array")
        return

    color_layers = resolved.get("color_layers") if resolved else None
    seen_recipes: Dict[Tuple[int, ...], int] = {}

    for e_idx, entry in enumerate(entries):
        e_path = f"{s_path}.entries[{e_idx}]"
        if not _is_mapping(entry):
            c.must("S-entry", e_path, "entry must be object / map")
            continue

        # V20: lab present, 3 finite reals, array form in JSON /
        #      array-or-12-byte-bin in msgpack.
        _validate_lab(entry.get("lab") if "lab" in entry else None,
                      "lab" in entry, e_path, c)

        # V9 / V10 / V15 / V25: recipe
        recipe_parsed = _validate_recipe(
            entry.get("recipe") if "recipe" in entry else None,
            "recipe" in entry,
            e_path,
            c,
            palette_len,
            color_layers,
        )

        if recipe_parsed is not None:
            # V12: same-section recipe deduplication.
            if recipe_parsed in seen_recipes:
                other = seen_recipes[recipe_parsed]
                c.should(
                    "V12",
                    e_path,
                    f"recipe duplicates entry[{other}]",
                )
            else:
                seen_recipes[recipe_parsed] = e_idx


def _validate_lab(
    lab: Any,
    present: bool,
    e_path: str,
    c: _Collector,
) -> None:
    if not present:
        c.must("V20", f"{e_path}.lab", "lab is required")
        return
    # Accept: list/tuple of 3 numbers (JSON + msgpack array form),
    # or bytes of length 12 (msgpack bin form; already unpacked to the
    # codec layer that feeds us lists normally - we still tolerate bytes
    # here to keep the validator usable by raw msgpack trees).
    if isinstance(lab, (bytes, bytearray, memoryview)):
        if len(lab) != 12:
            c.must(
                "V25",
                f"{e_path}.lab",
                f"lab bin must be exactly 12 bytes (got {len(lab)})",
            )
            return
        # V20: unpack the 12-byte bin (3 x float32 LE) and check finite.
        # Without this, a NaN/Inf-carrying bin would pass validate() but
        # crash at _build_color_db time, causing CLI/loader semantic split.
        values = struct.unpack("<fff", bytes(lab))
        for i, v in enumerate(values):
            if not math.isfinite(v):
                c.must(
                    "V20",
                    f"{e_path}.lab[{i}]",
                    f"lab[{i}] must be a finite number (got {v!r} in bin form)",
                )
        return
    if not _is_sequence(lab) or len(lab) != 3:
        c.must("V20", f"{e_path}.lab", "lab must be an array of length 3")
        return
    for i, v in enumerate(lab):
        if not _is_number(v) or not math.isfinite(float(v)):
            c.must(
                "V20",
                f"{e_path}.lab[{i}]",
                f"lab[{i}] must be a finite number (got {v!r})",
            )
            return


def _validate_recipe(
    recipe: Any,
    present: bool,
    e_path: str,
    c: _Collector,
    palette_len: int,
    color_layers: Optional[int],
) -> Optional[Tuple[int, ...]]:
    if not present:
        c.must("V9", f"{e_path}.recipe", "recipe is required")
        return None

    # V15: accept array or bin (bytes).
    is_bin = isinstance(recipe, (bytes, bytearray, memoryview))
    if is_bin:
        indices: Tuple[int, ...] = tuple(recipe)
    elif _is_sequence(recipe):
        vals: List[int] = []
        for i, v in enumerate(recipe):
            # V10 (first half): every element MUST be an integer in
            # [0, 255]. The "< palette_len or == 255" range check is the
            # second half, handled below.
            if not _is_int(v) or v < 0 or v > 255:
                c.must(
                    "V10",
                    f"{e_path}.recipe[{i}]",
                    f"recipe element must be uint8 integer in [0, 255] (got {v!r})",
                )
                return None
            vals.append(v)
        indices = tuple(vals)
    else:
        c.must("V15", f"{e_path}.recipe", "recipe must be array or bin")
        return None

    # Length check: bin-form errors are classified under V25
    # (MessagePack-specific byte length), array-form errors under V9.
    if color_layers is not None and len(indices) != color_layers:
        if is_bin:
            c.must(
                "V25",
                f"{e_path}.recipe",
                f"recipe bin length {len(indices)} != resolved color_layers {color_layers}",
            )
        else:
            c.must(
                "V9",
                f"{e_path}.recipe",
                f"recipe length {len(indices)} != resolved color_layers {color_layers}",
            )
        return None

    # V10: each index < palette_len or == AIR_INDEX.
    for i, v in enumerate(indices):
        if v == AIR_INDEX:
            continue
        if palette_len > 0 and v >= palette_len:
            c.must(
                "V10",
                f"{e_path}.recipe[{i}]",
                f"recipe index {v} >= palette length {palette_len} and != 255 (Air)",
            )
            return None

    return indices


# --------------------------------------------------------------------------
# §3.5 preset multiset matching (not a validator rule; convenience helper).
# --------------------------------------------------------------------------


def match_preset(palette: Sequence[Any]) -> Set[str]:
    """Return the set of standard preset names whose channel_class
    multiset exactly matches ``palette``'s multiset.

    The ``all`` preset is always included as it means "no filter" per
    spec §3.5.

    ``palette`` may be a sequence of :class:`PaletteEntry`, dicts, or
    bare :class:`str` values; only the ``channel_class`` is consulted.
    Items that do not expose a recognizable ``channel_class`` field are
    silently skipped (the caller is expected to have validated first).
    """
    classes: List[str] = []
    for item in palette:
        if _is_string(item):
            cc = item
        elif _is_mapping(item):
            cc = item.get("channel_class")
        else:
            cc = getattr(item, "channel_class", None)
        if _is_string(cc):
            classes.append(cc)

    matches: Set[str] = {"all"}
    if not classes:
        return matches
    palette_multiset = tuple(sorted(classes))
    for name, preset in STANDARD_PRESETS.items():
        if palette_multiset == preset:
            matches.add(name)
    return matches
