"""
Constants for the ColorDB spec v1 reference implementation.

All values here map 1-to-1 to ``docs/colordb-spec.md``; this module is
reference-only and is not meant for runtime consumers.

Sections referenced:
- SCHEMA_VERSION  -> spec §2
- MAX_PALETTE_SIZE / AIR_INDEX -> spec §3.2, §9.2
- CHANNEL_CLASSES -> spec §3.3 (52 items, PascalCase, case-sensitive)
- STANDARD_PRESETS -> spec §3.5 (multiset equality)
- UNICODE_WHITE_SPACE -> spec §4.1 step 3 (UCD 15.1 PropList.txt White_Space=Yes, 25 codepoints)
- DEFAULT_IGNORABLE_CODE_POINTS -> spec §11 V17 (UCD 15.1 DerivedCoreProperties.txt Default_Ignorable_Code_Point)
- RECOMMENDED_HEX -> spec Appendix B (informative, MUST NOT enter §11 validation)
- CHANNEL_CLASS_I18N_EN_ZH -> spec Appendix C (informative, MUST NOT enter §11 validation)

Unicode version lock: UCD 15.1 (Unicode 15.1.0, released 2023-09-12).
If Unicode upgrades materially change these derived properties, bump
the spec version in lockstep and regenerate the hardcoded tables.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Mapping, Tuple

SCHEMA_VERSION: int = 1
MAX_PALETTE_SIZE: int = 255
AIR_INDEX: int = 255

# --------------------------------------------------------------------------
# §3.3 channel_class enumeration (52 values, PascalCase, case-sensitive).
# --------------------------------------------------------------------------

_BASIC_HUES: Tuple[str, ...] = (
    "Red", "Orange", "Yellow", "Green", "Cyan", "Blue", "Purple",
    "Magenta", "Pink", "Brown", "White", "Gray", "Black",
)  # 13

_LIGHT_DARK: Tuple[str, ...] = (
    "LightRed", "DarkRed",
    "LightOrange", "DarkOrange",
    "LightYellow", "DarkYellow",
    "LightGreen", "DarkGreen",
    "LightBlue", "DarkBlue",
    "LightPurple", "DarkPurple",
    "LightBrown", "DarkBrown",
    "LightGray", "DarkGray",
)  # 16

_NAMED_COLORS: Tuple[str, ...] = (
    "Beige", "Ivory", "Cream", "Navy", "Teal", "Turquoise",
    "Olive", "Khaki", "Mint", "Coral", "Lavender",
    "Maroon", "Burgundy",
)  # 13

_METALLIC: Tuple[str, ...] = ("Gold", "Silver", "Bronze", "Copper")  # 4

_SPECIAL_APPEARANCE: Tuple[str, ...] = (
    "Transparent", "Translucent", "Fluorescent", "Glow", "Multicolor",
)  # 5

_FALLBACK: Tuple[str, ...] = ("Other",)  # 1

CHANNEL_CLASSES: Tuple[str, ...] = (
    _BASIC_HUES + _LIGHT_DARK + _NAMED_COLORS
    + _METALLIC + _SPECIAL_APPEARANCE + _FALLBACK
)
assert len(CHANNEL_CLASSES) == 52, (
    "CHANNEL_CLASSES must total 52 items per spec §3.3"
)
CHANNEL_CLASSES_SET: FrozenSet[str] = frozenset(CHANNEL_CLASSES)

# --------------------------------------------------------------------------
# §3.5 standard preset -> channel_class multiset mapping.
# ``all`` intentionally matches any palette and is not subject to multiset
# equality; consumers short-circuit it. Stored as sorted tuples so callers
# can compare multiset equality by sorting the palette's channel_class
# sequence.
# --------------------------------------------------------------------------

def _multiset(*items: str) -> Tuple[str, ...]:
    return tuple(sorted(items))


STANDARD_PRESETS: Mapping[str, Tuple[str, ...]] = {
    "CMYK": _multiset("Cyan", "Magenta", "Yellow", "Black"),
    "CMYW": _multiset("Cyan", "Magenta", "Yellow", "White"),
    "CMYKW": _multiset("Cyan", "Magenta", "Yellow", "Black", "White"),
    "RYBW": _multiset("Red", "Yellow", "Blue", "White"),
    "RGBW": _multiset("Red", "Green", "Blue", "White"),
    "RYB": _multiset("Red", "Yellow", "Blue"),
    "RGB": _multiset("Red", "Green", "Blue"),
}

# --------------------------------------------------------------------------
# §4.1 step 3: Unicode White_Space property codepoints.
# Source: UCD 15.1 PropList.txt (25 codepoints with White_Space=Yes).
# Hardcoded; implementations SHOULD NOT rely on ``str.isspace()`` per
# spec §4.1 note (Python's ``str.isspace()`` is NOT UCD-faithful).
# --------------------------------------------------------------------------

UNICODE_WHITE_SPACE: FrozenSet[int] = frozenset(
    {
        0x0009,  # CHARACTER TABULATION
        0x000A,  # LINE FEED
        0x000B,  # LINE TABULATION
        0x000C,  # FORM FEED
        0x000D,  # CARRIAGE RETURN
        0x0020,  # SPACE
        0x0085,  # NEXT LINE
        0x00A0,  # NO-BREAK SPACE
        0x1680,  # OGHAM SPACE MARK
        0x2000,  # EN QUAD
        0x2001,  # EM QUAD
        0x2002,  # EN SPACE
        0x2003,  # EM SPACE
        0x2004,  # THREE-PER-EM SPACE
        0x2005,  # FOUR-PER-EM SPACE
        0x2006,  # SIX-PER-EM SPACE
        0x2007,  # FIGURE SPACE
        0x2008,  # PUNCTUATION SPACE
        0x2009,  # THIN SPACE
        0x200A,  # HAIR SPACE
        0x2028,  # LINE SEPARATOR
        0x2029,  # PARAGRAPH SEPARATOR
        0x202F,  # NARROW NO-BREAK SPACE
        0x205F,  # MEDIUM MATHEMATICAL SPACE
        0x3000,  # IDEOGRAPHIC SPACE
    }
)
assert len(UNICODE_WHITE_SPACE) == 25, (
    "UNICODE_WHITE_SPACE must total 25 codepoints per UCD 15.1 PropList.txt"
)

# --------------------------------------------------------------------------
# §11 V17 (strengthened): Default_Ignorable_Code_Point set.
# Source: UCD 15.1 DerivedCoreProperties.txt.
#
# V17 MUST: display_name normalized per §4.1 MUST retain at least one
# codepoint that is neither White_Space (already stripped by normalize)
# nor Default_Ignorable_Code_Point. This set is consulted for the
# latter half of that check.
#
# Implementation: ranges expanded to a frozenset at import time. Total
# ~4174 codepoints (dominated by the U+E0000..U+E0FFF TAG / Variation
# Selectors Supplement block). Memory footprint is small; lookup is
# O(1).
# --------------------------------------------------------------------------

_DEFAULT_IGNORABLE_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x00AD, 0x00AD),   # SOFT HYPHEN
    (0x034F, 0x034F),   # COMBINING GRAPHEME JOINER
    (0x061C, 0x061C),   # ARABIC LETTER MARK
    (0x115F, 0x1160),   # HANGUL CHOSEONG / JUNGSEONG FILLER
    (0x17B4, 0x17B5),   # KHMER VOWEL INHERENT AQ / AA
    (0x180B, 0x180F),   # MONGOLIAN FREE VARIATION SELECTOR ONE..FOUR, VOWEL SEPARATOR
    (0x200B, 0x200F),   # ZWSP..RLM
    (0x202A, 0x202E),   # LRE..RLO
    (0x2060, 0x206F),   # WORD JOINER..NOMINAL DIGIT SHAPES
    (0x3164, 0x3164),   # HANGUL FILLER
    (0xFE00, 0xFE0F),   # VARIATION SELECTOR-1..16
    (0xFEFF, 0xFEFF),   # ZERO WIDTH NO-BREAK SPACE (BOM)
    (0xFFA0, 0xFFA0),   # HALFWIDTH HANGUL FILLER
    (0xFFF0, 0xFFF8),   # reserved (property-set in UCD)
    (0x1BCA0, 0x1BCA3), # SHORTHAND FORMAT LETTER OVERLAP..UP STEP
    (0x1D173, 0x1D17A), # MUSICAL SYMBOL BEGIN BEAM..END PHRASE
    (0xE0000, 0xE0FFF), # TAG block + Variation Selectors Supplement + reserved
)

DEFAULT_IGNORABLE_CODE_POINTS: FrozenSet[int] = frozenset(
    cp
    for lo, hi in _DEFAULT_IGNORABLE_RANGES
    for cp in range(lo, hi + 1)
)
assert 0x200B in DEFAULT_IGNORABLE_CODE_POINTS   # ZWSP sanity
assert 0x00AD in DEFAULT_IGNORABLE_CODE_POINTS   # SOFT HYPHEN sanity
assert 0x0041 not in DEFAULT_IGNORABLE_CODE_POINTS  # 'A' must NOT be flagged
assert 0xE0100 in DEFAULT_IGNORABLE_CODE_POINTS  # VS-17 sanity


def is_default_ignorable(cp: int) -> bool:
    """Return True iff ``cp`` has UCD 15.1 Default_Ignorable_Code_Point property."""
    return cp in DEFAULT_IGNORABLE_CODE_POINTS

# --------------------------------------------------------------------------
# Appendix B (informative): recommended hex color per channel_class.
# MUST NOT enter §11 validation.
# --------------------------------------------------------------------------

RECOMMENDED_HEX: Mapping[str, str] = {
    # B.1 basic hues
    "Red": "#d62828",
    "Orange": "#f77f00",
    "Yellow": "#fcbf49",
    "Green": "#2a9d8f",
    "Cyan": "#00b4d8",
    "Blue": "#1d3557",
    "Purple": "#6a4c93",
    "Magenta": "#d81159",
    "Pink": "#ffafcc",
    "Brown": "#6f4e37",
    "White": "#ffffff",
    "Gray": "#808080",
    "Black": "#000000",
    # B.2 light/dark variants
    "LightRed": "#ff6b6b", "DarkRed": "#8b0000",
    "LightOrange": "#ffa94d", "DarkOrange": "#bf4500",
    "LightYellow": "#fff59d", "DarkYellow": "#c9a227",
    "LightGreen": "#a8e6cf", "DarkGreen": "#003f00",
    "LightBlue": "#a0d2eb", "DarkBlue": "#001f54",
    "LightPurple": "#c5a3cf", "DarkPurple": "#4a1e5f",
    "LightBrown": "#b08968", "DarkBrown": "#3d251a",
    "LightGray": "#c0c0c0", "DarkGray": "#404040",
    # B.3 named colors
    "Beige": "#f5f5dc", "Ivory": "#fffff0", "Cream": "#fffdd0",
    "Navy": "#001f3f", "Teal": "#008080", "Turquoise": "#40e0d0",
    "Olive": "#808000", "Khaki": "#c3b091", "Mint": "#b5e4c0",
    "Coral": "#ff7f50", "Lavender": "#b497bd",
    "Maroon": "#800000", "Burgundy": "#7b1f3c",
    # B.4 metallic
    "Gold": "#d4af37", "Silver": "#b8c0c6",
    "Bronze": "#cd7f32", "Copper": "#b87333",
    # B.5 special appearance
    "Transparent": "#f0f0f0", "Translucent": "#e0e0e0",
    "Fluorescent": "#ff00ff", "Glow": "#ccffcc", "Multicolor": "#c0c0c0",
    # B.6 fallback
    "Other": "#808080",
}
assert set(RECOMMENDED_HEX.keys()) == CHANNEL_CLASSES_SET

# --------------------------------------------------------------------------
# Appendix C (informative): en-US / zh-CN UI translation suggestions.
# MUST NOT enter §11 validation.
# --------------------------------------------------------------------------

CHANNEL_CLASS_I18N_EN_ZH: Mapping[str, Mapping[str, str]] = {
    # C.1 basic hues
    "Red": {"en-US": "Red", "zh-CN": "红"},
    "Orange": {"en-US": "Orange", "zh-CN": "橙"},
    "Yellow": {"en-US": "Yellow", "zh-CN": "黄"},
    "Green": {"en-US": "Green", "zh-CN": "绿"},
    "Cyan": {"en-US": "Cyan", "zh-CN": "青"},
    "Blue": {"en-US": "Blue", "zh-CN": "蓝"},
    "Purple": {"en-US": "Purple", "zh-CN": "紫"},
    "Magenta": {"en-US": "Magenta", "zh-CN": "品红"},
    "Pink": {"en-US": "Pink", "zh-CN": "粉"},
    "Brown": {"en-US": "Brown", "zh-CN": "棕"},
    "White": {"en-US": "White", "zh-CN": "白"},
    "Gray": {"en-US": "Gray", "zh-CN": "灰"},
    "Black": {"en-US": "Black", "zh-CN": "黑"},
    # C.2 light/dark variants
    "LightRed": {"en-US": "Light Red", "zh-CN": "浅红"},
    "DarkRed": {"en-US": "Dark Red", "zh-CN": "深红"},
    "LightOrange": {"en-US": "Light Orange", "zh-CN": "浅橙"},
    "DarkOrange": {"en-US": "Dark Orange", "zh-CN": "深橙"},
    "LightYellow": {"en-US": "Light Yellow", "zh-CN": "浅黄"},
    "DarkYellow": {"en-US": "Dark Yellow", "zh-CN": "深黄"},
    "LightGreen": {"en-US": "Light Green", "zh-CN": "浅绿"},
    "DarkGreen": {"en-US": "Dark Green", "zh-CN": "深绿"},
    "LightBlue": {"en-US": "Light Blue", "zh-CN": "浅蓝"},
    "DarkBlue": {"en-US": "Dark Blue", "zh-CN": "深蓝"},
    "LightPurple": {"en-US": "Light Purple", "zh-CN": "浅紫"},
    "DarkPurple": {"en-US": "Dark Purple", "zh-CN": "深紫"},
    "LightBrown": {"en-US": "Light Brown", "zh-CN": "浅棕"},
    "DarkBrown": {"en-US": "Dark Brown", "zh-CN": "深棕"},
    "LightGray": {"en-US": "Light Gray", "zh-CN": "浅灰"},
    "DarkGray": {"en-US": "Dark Gray", "zh-CN": "深灰"},
    # C.3 named colors
    "Beige": {"en-US": "Beige", "zh-CN": "米色"},
    "Ivory": {"en-US": "Ivory", "zh-CN": "象牙色"},
    "Cream": {"en-US": "Cream", "zh-CN": "奶油色"},
    "Navy": {"en-US": "Navy", "zh-CN": "藏青"},
    "Teal": {"en-US": "Teal", "zh-CN": "蓝绿"},
    "Turquoise": {"en-US": "Turquoise", "zh-CN": "绿松石色"},
    "Olive": {"en-US": "Olive", "zh-CN": "橄榄绿"},
    "Khaki": {"en-US": "Khaki", "zh-CN": "卡其色"},
    "Mint": {"en-US": "Mint", "zh-CN": "薄荷绿"},
    "Coral": {"en-US": "Coral", "zh-CN": "珊瑚色"},
    "Lavender": {"en-US": "Lavender", "zh-CN": "薰衣草紫"},
    "Maroon": {"en-US": "Maroon", "zh-CN": "栗红"},
    "Burgundy": {"en-US": "Burgundy", "zh-CN": "酒红"},
    # C.4 metallic
    "Gold": {"en-US": "Gold", "zh-CN": "金"},
    "Silver": {"en-US": "Silver", "zh-CN": "银"},
    "Bronze": {"en-US": "Bronze", "zh-CN": "青铜"},
    "Copper": {"en-US": "Copper", "zh-CN": "铜"},
    # C.5 special appearance
    "Transparent": {"en-US": "Transparent", "zh-CN": "透明"},
    "Translucent": {"en-US": "Translucent", "zh-CN": "半透明"},
    "Fluorescent": {"en-US": "Fluorescent", "zh-CN": "荧光"},
    "Glow": {"en-US": "Glow-in-the-Dark", "zh-CN": "夜光"},
    "Multicolor": {"en-US": "Multicolor", "zh-CN": "多色"},
    # C.6 fallback
    "Other": {"en-US": "Other", "zh-CN": "其他"},
}
assert set(CHANNEL_CLASS_I18N_EN_ZH.keys()) == CHANNEL_CLASSES_SET

# --------------------------------------------------------------------------
# Regex (string form) for hex_color validation (#RRGGBB, case-insensitive).
# Spec §3.1: "格式 `#RRGGBB`".
# --------------------------------------------------------------------------
HEX_COLOR_PATTERN: str = r"^#[0-9A-Fa-f]{6}$"

# --------------------------------------------------------------------------
# Section type enumeration (spec §8.3).
# --------------------------------------------------------------------------
SECTION_TYPES: Tuple[str, ...] = ("measured", "predicted")

__all__ = [
    "SCHEMA_VERSION",
    "MAX_PALETTE_SIZE",
    "AIR_INDEX",
    "CHANNEL_CLASSES",
    "CHANNEL_CLASSES_SET",
    "STANDARD_PRESETS",
    "UNICODE_WHITE_SPACE",
    "DEFAULT_IGNORABLE_CODE_POINTS",
    "is_default_ignorable",
    "RECOMMENDED_HEX",
    "CHANNEL_CLASS_I18N_EN_ZH",
    "HEX_COLOR_PATTERN",
    "SECTION_TYPES",
]
