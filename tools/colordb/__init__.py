"""
ColorDB spec v1 -- Python reference implementation.

This package is a **reference implementation** that maps one-to-one to
``docs/colordb-spec.md``. It is not a runtime component of any
ChromaPrint3D service; its purpose is to provide a single, rigorous
interpretation of the spec that other implementations (C++, TypeScript
and Python consumers) can cross-check against.

Sections of the spec that are **not** implemented in this package are:

- §3.4 (channel_class classification decision order): an authoring
  guideline for dataset producers, not loader behaviour.
- §12.1 (persisted-artifact locale sensitivity): consumer architecture
  guidance; no code path is prescribed.
- §12.2 (API request field conventions): non-normative suggestion for
  consumers that expose a channel-filtering API.

All merge operations are palette-only (§4.3); this package does not
attempt to merge full :class:`ColorDB` documents.

Public API surface is re-exported below.
"""

from __future__ import annotations

from .codec import (
    detect_encoding,
    dump_bytes,
    dump_doc,
    dump_path,
    load_bytes,
    load_doc,
    load_path,
)
from .constants import (
    AIR_INDEX,
    CHANNEL_CLASS_I18N_EN_ZH,
    CHANNEL_CLASSES,
    HEX_COLOR_PATTERN,
    MAX_PALETTE_SIZE,
    RECOMMENDED_HEX,
    SCHEMA_VERSION,
    SECTION_TYPES,
    STANDARD_PRESETS,
)
from .locale_tag import fallback, fallback_chain, is_valid_bcp47, spec_canonicalize
from .merge import ColorDBMergeError, merge_palettes
from .model import (
    ChannelClass,
    ColorDB,
    Defaults,
    Entry,
    PaletteEntry,
    ResolvedConfig,
    Section,
    SectionType,
)
from .normalize import normalize
from .validator import (
    ColorDBValidationError,
    ValidationIssue,
    ValidationReport,
    match_preset,
    validate,
)

# Short aliases matching the plan's public names.
load = load_bytes
dump = dump_bytes

__all__ = [
    # Parse / serialize
    "load",
    "dump",
    "load_bytes",
    "load_doc",
    "load_path",
    "dump_bytes",
    "dump_doc",
    "dump_path",
    "detect_encoding",
    # Validate / merge / i18n
    "validate",
    "merge_palettes",
    "fallback",
    "fallback_chain",
    # Utilities
    "normalize",
    "spec_canonicalize",
    "is_valid_bcp47",
    "match_preset",
    # Types
    "ChannelClass",
    "SectionType",
    "ColorDB",
    "PaletteEntry",
    "Defaults",
    "Section",
    "Entry",
    "ResolvedConfig",
    "ValidationIssue",
    "ValidationReport",
    # Exceptions
    "ColorDBValidationError",
    "ColorDBMergeError",
    # Constants
    "SCHEMA_VERSION",
    "AIR_INDEX",
    "MAX_PALETTE_SIZE",
    "STANDARD_PRESETS",
    "CHANNEL_CLASSES",
    "SECTION_TYPES",
    "HEX_COLOR_PATTERN",
    "RECOMMENDED_HEX",
    "CHANNEL_CLASS_I18N_EN_ZH",
]
