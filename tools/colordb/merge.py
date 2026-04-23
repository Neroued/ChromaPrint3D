"""
Cross-ColorDB palette merge per spec §4.3 (fail-hard).

The merge covers **only the palette dimension**. The spec does not
define merge semantics for ``name`` / ``vendor`` / ``material_type`` /
``defaults`` / ``meta`` / ``sections``, so this module does not pretend
to merge full :class:`ColorDB` objects; it takes palette sequences and
returns a merged palette sequence.

Spec §4.3 SHOULD treatment (C3): for fields outside the palette table,
callers MUST govern merge semantics themselves (keep first / keep last /
reject conflicts / ...). This module deliberately refuses to silently
merge those fields so that regressions caused by implicit merging
surface as a type error (missing arguments) rather than hidden data
corruption.

Merge key (§4.3):

    (normalize(display_name), normalize(material))

Hard-conflict fields (any mismatch raises :class:`ColorDBMergeError`):

- ``channel_class``
- ``hex_color``
- ``display_name`` (original literal, not normalized)
- ``material`` (original literal; ``"PLA Basic"`` vs ``"pla basic"``
  share a merge key but still fail hard)
- ``display_name_localized`` (compared after spec-canonicalizing keys;
  same canonical key with different values fails; disjoint keys are
  unioned)

Exact duplicates (all fields identical, including the spec-canonical
localized map) are silently deduplicated.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .locale_tag import spec_canonicalize
from .model import PaletteEntry
from .normalize import normalize

__all__ = ["ColorDBMergeError", "merge_palettes"]


class ColorDBMergeError(Exception):
    """Raised when two palette entries with identical merge keys conflict."""


def _canonicalize_localized(
    localized: Optional[Mapping[str, str]],
) -> Dict[str, str]:
    """Return a dict keyed by spec-canonical BCP 47 tag.

    If two input keys canonicalize to the same tag with different
    values, raises :class:`ColorDBMergeError` since that is a V19
    violation and should already have been rejected by the loader;
    we double-check here in case callers merge unvalidated data.
    """
    if not localized:
        return {}
    out: Dict[str, str] = {}
    for raw_key, value in localized.items():
        canonical = spec_canonicalize(raw_key)
        if canonical in out and out[canonical] != value:
            raise ColorDBMergeError(
                f"display_name_localized has duplicate canonical key "
                f"{canonical!r} with different values (V19 violation in input)"
            )
        out[canonical] = value
    return out


def _merge_localized(
    left: Mapping[str, str],
    right: Mapping[str, str],
    where: str,
) -> Dict[str, str]:
    merged: Dict[str, str] = dict(left)
    for key, value in right.items():
        if key in merged:
            if merged[key] != value:
                raise ColorDBMergeError(
                    f"display_name_localized[{key!r}] conflict at {where}: "
                    f"{merged[key]!r} vs {value!r}"
                )
        else:
            merged[key] = value
    return merged


def _entry_key(entry: PaletteEntry) -> Tuple[str, str]:
    return (normalize(entry.display_name), normalize(entry.material))


def _entries_equal(a: PaletteEntry, b: PaletteEntry) -> bool:
    return (
        a.channel_class == b.channel_class
        and a.display_name == b.display_name
        and a.material == b.material
        and a.hex_color == b.hex_color
        and _canonicalize_localized(a.display_name_localized)
        == _canonicalize_localized(b.display_name_localized)
    )


def merge_palettes(
    palettes: Sequence[Sequence[PaletteEntry]],
) -> List[PaletteEntry]:
    """Return the merged palette from multiple palette sequences.

    Entries with the same merge key are collapsed into a single entry:

    - If all non-localized fields match and the spec-canonical localized
      maps are compatible (disjoint, or identical on shared keys), the
      result contains one entry whose ``display_name_localized`` is the
      union of the inputs' maps, re-keyed to the spec-canonical form.
    - Any conflict on the 5 spec-listed fields raises
      :class:`ColorDBMergeError` without performing partial output.

    The returned list preserves first-seen order across the input
    palettes.
    """
    merged: Dict[Tuple[str, str], PaletteEntry] = {}
    order: List[Tuple[str, str]] = []

    for p_idx, palette in enumerate(palettes):
        for e_idx, entry in enumerate(palette):
            where = f"palette[{p_idx}].entry[{e_idx}] ({entry.display_name!r}/{entry.material!r})"
            key = _entry_key(entry)
            if key not in merged:
                # Canonicalize the localized map on first insertion so
                # that subsequent comparisons are apples-to-apples.
                canonical_loc = _canonicalize_localized(entry.display_name_localized)
                merged[key] = replace(entry, display_name_localized=canonical_loc)
                order.append(key)
                continue

            existing = merged[key]
            # Exact duplicate fast path.
            if _entries_equal(existing, entry):
                continue

            # Hard-conflict fields.
            if existing.channel_class != entry.channel_class:
                raise ColorDBMergeError(
                    f"channel_class conflict at {where}: "
                    f"{existing.channel_class!r} vs {entry.channel_class!r}"
                )
            if existing.hex_color != entry.hex_color:
                raise ColorDBMergeError(
                    f"hex_color conflict at {where}: "
                    f"{existing.hex_color!r} vs {entry.hex_color!r}"
                )
            if existing.display_name != entry.display_name:
                raise ColorDBMergeError(
                    f"display_name literal conflict at {where}: "
                    f"{existing.display_name!r} vs {entry.display_name!r}"
                )
            if existing.material != entry.material:
                raise ColorDBMergeError(
                    f"material literal conflict at {where}: "
                    f"{existing.material!r} vs {entry.material!r}"
                )

            # Localized map: union with same-key value check.
            new_loc = _canonicalize_localized(entry.display_name_localized)
            combined = _merge_localized(
                existing.display_name_localized, new_loc, where
            )
            merged[key] = replace(existing, display_name_localized=combined)

    return [merged[key] for key in order]
