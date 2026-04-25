#!/usr/bin/env python3
"""flush_reference.py — Reference Python implementation of the HSV-based
flush-volume formula used by ChromaPrint3D's `FlushVolumeCalculator`.

Mirrors `core/src/flush/flush_calculator.cpp::CalcFlushVolRgb` exactly,
which is itself derived from
`BambuStudio/src/libslic3r/FlushVolCalc.cpp::calc_flush_vol_rgb` (AGPL-3.0).

This is the **ground-truth oracle** for the C++ unit tests in
`core/tests/test_flush.cpp`. To recompute the expected values:

    python3 scripts/dev/flush_reference.py

Tolerance: tests allow ±1 mm³ to absorb the float (C++) vs double (Python)
last-bit difference.
"""

from __future__ import annotations

import math


def hex_to_rgb(s: str) -> tuple[int, int, int]:
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        raise ValueError(f"bad hex: {s!r}")
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Inputs in [0,1] gamma-encoded sRGB. Matches BBS RGB2HSV."""
    Cmax = max(r, g, b)
    Cmin = min(r, g, b)
    delta = Cmax - Cmin
    if abs(delta) < 0.001:
        h = 0.0
    elif Cmax == r:
        h = 60.0 * math.fmod((g - b) / delta, 6.0)
    elif Cmax == g:
        h = 60.0 * ((b - r) / delta + 2)
    else:
        h = 60.0 * ((r - g) / delta + 4)
    s = 0.0 if abs(Cmax) < 0.001 else delta / Cmax
    v = Cmax
    return h, s, v


def delta_hs_bbs(h1, s1, v1, h2, s2, v2) -> float:
    h1r = math.radians(h1)
    h2r = math.radians(h2)
    dx = math.cos(h1r) * s1 * v1 - math.cos(h2r) * s2 * v2
    dy = math.sin(h1r) * s1 * v1 - math.sin(h2r) * s2 * v2
    return min(1.2, math.sqrt(dx * dx + dy * dy))


def calc_flush_vol_rgb(src_rgb: tuple[int, int, int],
                        dst_rgb: tuple[int, int, int]) -> float:
    """BBS HSV-based formula. Returns mm³ (no min/max applied)."""
    sr, sg, sb = (c / 255.0 for c in src_rgb)
    dr, dg, db = (c / 255.0 for c in dst_rgb)
    h1, s1, v1 = rgb_to_hsv(sr, sg, sb)
    h2, s2, v2 = rgb_to_hsv(dr, dg, db)
    hs_dist = delta_hs_bbs(h1, s1, v1, h2, s2, v2)
    from_lumi = 0.30 * sr + 0.59 * sg + 0.11 * sb
    to_lumi = 0.30 * dr + 0.59 * dg + 0.11 * db
    if to_lumi >= from_lumi:
        lumi_flush = (to_lumi - from_lumi) ** 0.7 * 560.0
    else:
        lumi_flush = (from_lumi - to_lumi) * 80.0
        inter_v = 0.67 * v2 + 0.33 * v1
        hs_dist = min(inter_v, hs_dist)
    hs_flush = 230.0 * hs_dist
    # Triangle 3rd edge with 120° between hs_flush and lumi_flush.
    volume = math.sqrt(
        hs_flush * hs_flush + lumi_flush * lumi_flush
        - 2 * hs_flush * lumi_flush * math.cos(math.radians(120))
    )
    return max(volume, 60.0)


def calc_flush_vol(src_hex: str, dst_hex: str,
                    min_flush_volume: int = 0,
                    max_flush_volume: int = 900) -> int:
    """Top-level: parse hex, compute, add min, clamp at max.

    Mirrors C++ `FlushVolumeCalculator::Calc` exactly. Note: there is NO
    same-color short-circuit here — the caller (e.g. matrix-builder) is
    responsible for setting diagonal entries to zero. For src == dst the
    formula returns the floor `60 + min_flush_volume` (HSV formula min).
    """
    src = hex_to_rgb(src_hex)
    dst = hex_to_rgb(dst_hex)
    volume = calc_flush_vol_rgb(src, dst)
    return min(int(volume + min_flush_volume), max_flush_volume)


# ── Driver: print expected values for the C++ test fixtures ──────────────────


def main() -> None:
    cases = [
        ("#000000", "#FFFFFF"),  # black → white
        ("#FFFFFF", "#000000"),  # white → black
        ("#FFFFFF", "#FFFFFF"),  # same color
        ("#000000", "#000000"),
        ("#C12E1F", "#0086D6"),  # BBL red → cyan
        ("#0086D6", "#C12E1F"),  # cyan → red (asymmetric)
        ("#F4EE2A", "#000000"),  # BBL yellow → black
        ("#2850E0", "#46A8F9"),  # custom blue → light blue (varesa palette)
    ]
    print("=== HSV flush volume (mm³, integer truncation, no min_flush) ===")
    for src, dst in cases:
        v = calc_flush_vol(src, dst, min_flush_volume=0, max_flush_volume=900)
        print(f"  {src} -> {dst}: {v}")
    print()
    print("To regenerate test_flush.cpp expected values, copy the integers above.")


if __name__ == "__main__":
    main()
