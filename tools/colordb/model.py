"""
Typed in-memory model for a ColorDB document per spec §2-§9.

The dataclasses here are immutable value objects. Construction is
performed exclusively by the codec / validator layers; they never
implicitly run the §4.1 ``normalize`` on any field. Consumers must
import :func:`tools.colordb.normalize.normalize` and apply it
explicitly whenever they need the normalized form (e.g. uniqueness
keys, merge keys, fallback lookups on user-supplied names).

Unknown fields at all four levels (top-level / palette entry / section /
entry) are discarded during construction, which satisfies V13. The
round-trip guarantee is **semantic**: ``dump(load(x))`` produces an
equivalent object, not a byte-identical copy of the original file.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, NamedTuple, Optional, Tuple

__all__ = [
    "ChannelClass",
    "SectionType",
    "PaletteEntry",
    "Defaults",
    "Entry",
    "Section",
    "ColorDB",
    "ResolvedConfig",
]


class ChannelClass(str, Enum):
    """Spec §3.3: 52 PascalCase enumeration values.

    Subclassing :class:`str` means both ``ChannelClass.Red`` and the
    literal ``"Red"`` compare equal and serialize identically; this is
    convenient for reference implementations that round-trip through
    JSON.
    """

    # §3.3.1 basic hues (13)
    Red = "Red"
    Orange = "Orange"
    Yellow = "Yellow"
    Green = "Green"
    Cyan = "Cyan"
    Blue = "Blue"
    Purple = "Purple"
    Magenta = "Magenta"
    Pink = "Pink"
    Brown = "Brown"
    White = "White"
    Gray = "Gray"
    Black = "Black"
    # §3.3.2 light / dark variants (16)
    LightRed = "LightRed"
    DarkRed = "DarkRed"
    LightOrange = "LightOrange"
    DarkOrange = "DarkOrange"
    LightYellow = "LightYellow"
    DarkYellow = "DarkYellow"
    LightGreen = "LightGreen"
    DarkGreen = "DarkGreen"
    LightBlue = "LightBlue"
    DarkBlue = "DarkBlue"
    LightPurple = "LightPurple"
    DarkPurple = "DarkPurple"
    LightBrown = "LightBrown"
    DarkBrown = "DarkBrown"
    LightGray = "LightGray"
    DarkGray = "DarkGray"
    # §3.3.3 named colors (13)
    Beige = "Beige"
    Ivory = "Ivory"
    Cream = "Cream"
    Navy = "Navy"
    Teal = "Teal"
    Turquoise = "Turquoise"
    Olive = "Olive"
    Khaki = "Khaki"
    Mint = "Mint"
    Coral = "Coral"
    Lavender = "Lavender"
    Maroon = "Maroon"
    Burgundy = "Burgundy"
    # §3.3.4 metallic (4)
    Gold = "Gold"
    Silver = "Silver"
    Bronze = "Bronze"
    Copper = "Copper"
    # §3.3.5 special appearance (5)
    Transparent = "Transparent"
    Translucent = "Translucent"
    Fluorescent = "Fluorescent"
    Glow = "Glow"
    Multicolor = "Multicolor"
    # §3.3.6 fallback (1)
    Other = "Other"


class SectionType(str, Enum):
    """Spec §8.3 section ``type`` values."""

    measured = "measured"
    predicted = "predicted"


class ResolvedConfig(NamedTuple):
    """Spec §8.2 resolved config (all five print parameters)."""

    color_layers: int
    layer_height_mm: float
    line_width_mm: float
    base_layers: int
    base_channel_idx: int


@dataclass(frozen=True)
class PaletteEntry:
    """Spec §3.1 palette entry.

    Fields beyond those defined by the spec are intentionally not
    accepted; loaders drop unknown keys before constructing this type.
    """

    channel_class: str
    display_name: str
    material: str
    hex_color: Optional[str] = None
    display_name_localized: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Defaults:
    """Spec §6 defaults block."""

    color_layers: int
    layer_height_mm: float
    line_width_mm: float
    base_layers: int
    base_channel_idx: int


@dataclass(frozen=True)
class Entry:
    """Spec §9.1 entry.

    ``lab`` is a 3-tuple of finite floats (V20). ``recipe`` is a tuple
    of integers; the codec layer accepts both array and bin
    representations (V15) and normalizes the value to a tuple here.
    """

    lab: Tuple[float, float, float]
    recipe: Tuple[int, ...]

    def __post_init__(self) -> None:  # pragma: no cover - defensive
        # Sanity check; validator.py does the user-facing MUST checks.
        if len(self.lab) != 3 or not all(math.isfinite(v) for v in self.lab):
            raise ValueError("Entry.lab must be 3 finite floats")


@dataclass(frozen=True)
class Section:
    """Spec §8.1 section.

    Fields left unset mean "inherit from defaults" per §8.2. The
    :meth:`resolved_config` method applies that inheritance.
    """

    type: str
    entries: Tuple[Entry, ...] = ()
    color_layers: Optional[int] = None
    layer_height_mm: Optional[float] = None
    line_width_mm: Optional[float] = None
    base_layers: Optional[int] = None
    base_channel_idx: Optional[int] = None
    threshold: Optional[float] = None
    margin: Optional[float] = None

    def resolved_config(self, defaults: Defaults) -> ResolvedConfig:
        """Return the §8.2 resolved config with inherited defaults."""
        return ResolvedConfig(
            color_layers=(
                self.color_layers
                if self.color_layers is not None
                else defaults.color_layers
            ),
            layer_height_mm=(
                self.layer_height_mm
                if self.layer_height_mm is not None
                else defaults.layer_height_mm
            ),
            line_width_mm=(
                self.line_width_mm
                if self.line_width_mm is not None
                else defaults.line_width_mm
            ),
            base_layers=(
                self.base_layers
                if self.base_layers is not None
                else defaults.base_layers
            ),
            base_channel_idx=(
                self.base_channel_idx
                if self.base_channel_idx is not None
                else defaults.base_channel_idx
            ),
        )


@dataclass(frozen=True)
class ColorDB:
    """Spec §2 top-level document."""

    schema_version: int
    name: str
    palette: Tuple[PaletteEntry, ...]
    defaults: Defaults
    sections: Tuple[Section, ...]
    vendor: Optional[str] = None
    material_type: Optional[str] = None
    meta: Mapping[str, object] = field(default_factory=dict)
