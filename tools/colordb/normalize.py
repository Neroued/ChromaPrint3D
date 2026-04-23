"""
String normalization per spec §4.1.

``normalize(s)`` applies exactly three steps, in order:

1. Unicode NFC (``unicodedata.normalize("NFC", s)``).
2. Unicode Default Case Folding (``str.casefold()``).
3. Strip every codepoint whose Unicode ``White_Space`` property is ``Yes``
   (see :data:`tools.colordb.constants.UNICODE_WHITE_SPACE`).

Unicode version lock: the White_Space codepoint set is taken from
UCD 15.1 (see ``constants.py``). Python's :func:`str.isspace` is
**not** used because it is not UCD-faithful across minor releases.

Non-goals (must not be performed by this function):

- NFKC / NFKD compatibility folding.
- ASCII-only filtering.
- Stripping of non-``White_Space`` invisibles such as
  ``U+200B`` / ``U+200C`` / ``U+200D`` / ``U+00AD`` / ``U+FEFF``.
  Callers that need to enforce the V17 "at least one non-ignorable
  character" rule do so by consulting
  :data:`tools.colordb.constants.DEFAULT_IGNORABLE_CODE_POINTS` on
  the output of :func:`normalize`.
- Confusable / security-profile detection.
"""

from __future__ import annotations

import unicodedata

from .constants import UNICODE_WHITE_SPACE

__all__ = ["normalize"]


def normalize(s: str) -> str:
    """Return the spec §4.1 normalized form of ``s``.

    The input must be a :class:`str`; any other type raises :class:`TypeError`
    so that callers cannot accidentally pass bytes or ``None``.
    """
    if not isinstance(s, str):
        raise TypeError(
            f"normalize() expects str, got {type(s).__name__}"
        )
    nfc = unicodedata.normalize("NFC", s)
    folded = nfc.casefold()
    # Step 3: delete codepoints in the hardcoded White_Space set.
    return "".join(ch for ch in folded if ord(ch) not in UNICODE_WHITE_SPACE)
