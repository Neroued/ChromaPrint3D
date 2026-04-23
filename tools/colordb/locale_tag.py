"""
BCP 47 language tag utilities per spec §5.1.1 and §5.2.

This module intentionally does **not** depend on any locale platform API
(``babel`` / ``langcodes`` / ``Intl.getCanonicalLocales`` / ICU) because
those perform extra work that the spec forbids:

- substitution (``iw`` → ``he``)
- likely subtags (``zh-TW`` → ``zh-Hant-TW``)
- grandfathered / redundant tag expansion (``i-klingon`` → ``tlh``)
- extlang folding (``zh-cmn-Hans-CN`` → ``cmn-Hans-CN``)
- alias normalization (``root`` → ``und``)

``spec_canonicalize`` performs subtag-wise case regularization **only**,
classifying subtags by **position** (not by ad-hoc length heuristics).
``is_valid_bcp47`` performs strict grammar checks for V18.
``fallback_chain`` and ``fallback`` implement the §5.2 four-step lookup.
"""

from __future__ import annotations

from typing import List, Mapping, Optional, Sequence

__all__ = [
    "spec_canonicalize",
    "is_valid_bcp47",
    "fallback_chain",
    "fallback",
]


# --------------------------------------------------------------------------
# Private subtag predicates.
# --------------------------------------------------------------------------

def _is_alpha(sub: str) -> bool:
    return len(sub) > 0 and all("A" <= c <= "Z" or "a" <= c <= "z" for c in sub)


def _is_digit(sub: str) -> bool:
    return len(sub) > 0 and all("0" <= c <= "9" for c in sub)


def _is_alnum(sub: str) -> bool:
    return len(sub) > 0 and all(
        "A" <= c <= "Z" or "a" <= c <= "z" or "0" <= c <= "9" for c in sub
    )


def _title_case(sub: str) -> str:
    if not sub:
        return sub
    return sub[0].upper() + sub[1:].lower()


# --------------------------------------------------------------------------
# §5.1.1: spec_canonicalize
# --------------------------------------------------------------------------


def spec_canonicalize(tag: str) -> str:
    """Return the spec §5.1.1 subtag-wise case-regularized form of ``tag``.

    This is **not** RFC 5646 canonical form. It does not perform
    substitution, likely-subtag inference, grandfathered tag expansion,
    extlang folding, or alias normalization.

    The function intentionally does **not** validate the input. Callers
    that need validation should use :func:`is_valid_bcp47` separately.
    Inputs that violate BCP 47 grammar are still processed by position
    heuristics, preserving the "best effort case regularization" contract
    of §5.1.1 without adding hidden validation.
    """
    if not isinstance(tag, str):
        raise TypeError(
            f"spec_canonicalize() expects str, got {type(tag).__name__}"
        )
    if not tag:
        return ""

    parts: List[str] = tag.split("-")
    result: List[str] = []
    i = 0
    n = len(parts)

    # Primary language: the first subtag.
    result.append(parts[i].lower())
    i += 1

    # Up to 3 consecutive extlang subtags (3-letter alpha).
    ext_count = 0
    while i < n and ext_count < 3 and _is_alpha(parts[i]) and len(parts[i]) == 3:
        result.append(parts[i].lower())
        i += 1
        ext_count += 1

    # Optional script: 4-letter alpha -> Title Case.
    if i < n and _is_alpha(parts[i]) and len(parts[i]) == 4:
        result.append(_title_case(parts[i]))
        i += 1

    # Optional region: 2-letter alpha (upper) or 3-digit numeric (preserved).
    if i < n and (
        (_is_alpha(parts[i]) and len(parts[i]) == 2)
        or (_is_digit(parts[i]) and len(parts[i]) == 3)
    ):
        result.append(parts[i].upper() if _is_alpha(parts[i]) else parts[i])
        i += 1

    # Variants: 5-8 chars, or 4 chars starting with a digit. All lowercase.
    while i < n:
        sub = parts[i]
        is_variant = (5 <= len(sub) <= 8) or (
            len(sub) == 4 and sub[0:1].isdigit()
        )
        if not is_variant:
            break
        result.append(sub.lower())
        i += 1

    # Extensions (singleton 0-9/A-W/Y-Z, not 'x') and privateuse (singleton 'x').
    # All following subtags are lowercased, which handles extension subtags
    # and privateuse subtags uniformly and matches the spec's rule that
    # both groups are rendered in lowercase.
    while i < n:
        result.append(parts[i].lower())
        i += 1

    return "-".join(result)


# --------------------------------------------------------------------------
# V18: strict BCP 47 grammar validator.
#
# RFC 5646 defines: Language-Tag = langtag / privateuse / grandfathered
# Grandfathered tags (irregular + regular) are a closed, finite set.
# --------------------------------------------------------------------------

# Complete grandfathered tag list from RFC 5646 §2.1 (case-insensitive).
# These are accepted verbatim by is_valid_bcp47 because BCP 47 grammar
# explicitly includes them, and spec §5.1 says "MUST accept any tag that
# conforms to BCP 47 syntax". The spec §5.1.1 only forbids *expanding*
# them (e.g. i-klingon MUST NOT be replaced with tlh).
_GRANDFATHERED: frozenset[str] = frozenset(
    t.lower()
    for t in (
        # irregular
        "en-GB-oed",
        "i-ami",
        "i-bnn",
        "i-default",
        "i-enochian",
        "i-hak",
        "i-klingon",
        "i-lux",
        "i-mingo",
        "i-navajo",
        "i-pwn",
        "i-tao",
        "i-tay",
        "i-tsu",
        "sgn-BE-FR",
        "sgn-BE-NL",
        "sgn-CH-DE",
        # regular
        "art-lojban",
        "cel-gaulish",
        "no-bok",
        "no-nyn",
        "zh-guoyu",
        "zh-hakka",
        "zh-min",
        "zh-min-nan",
        "zh-xiang",
    )
)


def is_valid_bcp47(tag: str) -> bool:
    """Return ``True`` if ``tag`` conforms to BCP 47 (RFC 5646) syntax.

    This performs a strict grammar check focused on what spec V18
    requires: each key in ``display_name_localized`` must be a legal
    language tag.

    Accepted productions:

    - **langtag**: primary language (2-3 / 4 / 5-8 letter alpha) plus
      optional extlang, script, region, variant, extension, privateuse.
    - **privateuse**: ``x-`` then 1+ subtags of 1-8 alnum chars.
    - **grandfathered**: the closed set from RFC 5646 (``i-klingon``,
      ``en-GB-oed``, ``zh-guoyu``, etc.), matched case-insensitively.

    Spec §5.1.1 forbids *expanding* grandfathered tags but §5.1 says
    the loader **MUST** accept any syntactically valid BCP 47 tag.
    """
    if not isinstance(tag, str) or not tag:
        return False

    parts = tag.split("-")
    if any(not p for p in parts):
        return False
    if any(not (1 <= len(p) <= 8) for p in parts):
        return False
    if any(not _is_alnum(p) for p in parts):
        return False

    n = len(parts)
    i = 0

    # Privateuse-only tag ("x-foo" / "x-foo-bar").
    if parts[0].lower() == "x":
        if n < 2:
            return False
        # All subsequent subtags are privateuse; already length-validated.
        return True

    # Grandfathered tags: closed set from RFC 5646 §2.1.
    if tag.lower() in _GRANDFATHERED:
        return True

    # Primary language: 2*3ALPHA / 4ALPHA / 5*8ALPHA per BCP 47 ABNF.
    first = parts[0]
    if not _is_alpha(first):
        return False
    if not (2 <= len(first) <= 8):
        return False
    i += 1

    # Up to 3 extlang subtags (3-letter alpha).
    ext_count = 0
    while i < n and ext_count < 3 and _is_alpha(parts[i]) and len(parts[i]) == 3:
        i += 1
        ext_count += 1

    # Optional script (4-letter alpha).
    if i < n and _is_alpha(parts[i]) and len(parts[i]) == 4:
        i += 1

    # Optional region (2-letter alpha or 3-digit numeric).
    if i < n:
        sub = parts[i]
        if (_is_alpha(sub) and len(sub) == 2) or (_is_digit(sub) and len(sub) == 3):
            i += 1

    # Variants (5-8 alnum, or 4-char alnum starting with a digit).
    while i < n:
        sub = parts[i]
        is_variant = (5 <= len(sub) <= 8) or (len(sub) == 4 and sub[0].isdigit())
        if not is_variant:
            break
        i += 1

    # Extension blocks: singleton (1 char, not 'x') then 1+ subtags 2-8 alnum.
    while i < n and len(parts[i]) == 1 and parts[i].lower() != "x":
        singleton = parts[i]
        if not _is_alnum(singleton):
            return False
        i += 1
        count = 0
        while i < n and 2 <= len(parts[i]) <= 8 and len(parts[i]) != 1:
            i += 1
            count += 1
            # Stop the extension block when the next subtag is a singleton.
            if i < n and len(parts[i]) == 1:
                break
        if count == 0:
            return False

    # Optional trailing privateuse block: 'x' then 1+ subtags (length 1-8).
    if i < n and parts[i].lower() == "x":
        i += 1
        if i >= n:
            return False
        # All remaining subtags are privateuse (already alnum & length-checked).
        i = n

    return i == n


# --------------------------------------------------------------------------
# §5.2: fallback chain and selector.
# --------------------------------------------------------------------------


def fallback_chain(locale: str) -> List[str]:
    """Return the §5.2 language-lookup chain for ``locale``.

    The input is first run through :func:`spec_canonicalize`, then
    right-to-left subtag trimming produces successively shorter tags until
    only the primary language remains. Extension and privateuse segments
    are treated as regular subtags and get trimmed off normally; this
    matches the RFC 4647 Lookup contract used by §5.2.

    The returned list is ordered from most specific to least specific and
    contains at least one element (the canonicalized input). An empty
    string yields an empty list.
    """
    if not isinstance(locale, str):
        raise TypeError(
            f"fallback_chain() expects str, got {type(locale).__name__}"
        )
    if not locale:
        return []
    canonical = spec_canonicalize(locale)
    parts = canonical.split("-")
    chain: List[str] = []
    while parts:
        chain.append("-".join(parts))
        parts.pop()
    return chain


def fallback(
    display_name: str,
    display_name_localized: Optional[Mapping[str, str]],
    user_locale: str,
) -> str:
    """Return the §5.2 localized display string for a palette entry.

    ``display_name_localized`` keys are assumed to already be in
    spec-canonical form (loaders are required to canonicalize on read).
    If they are not, this function will still canonicalize the user
    locale and attempt lookup; callers may canonicalize the map
    themselves before calling.

    Steps (spec §5.2):

    1. spec-canonicalize ``user_locale``.
    2. exact match in ``display_name_localized``.
    3. strip rightmost subtags and retry until only the primary language
       remains.
    4. fall back to ``display_name``.

    The function is read-only and never mutates the provided mapping.
    """
    if display_name_localized:
        for candidate in fallback_chain(user_locale):
            if candidate in display_name_localized:
                return display_name_localized[candidate]
    return display_name
