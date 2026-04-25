#pragma once

/// \file flush_calculator.h
/// \brief Color-driven filament flush-volume calculator.
///
/// Computes a "reasonable" purge volume (mm³) for a given (src, dst) hex
/// color pair using the HSV/luminance formula derived from BambuStudio's
/// `calc_flush_vol_rgb` (the fallback path that runs when their dataset
/// lookup misses). We deliberately do NOT replicate BBS's dataset-table
/// lookup or "dark→light ×1.3" boost: maintaining byte-level parity with
/// BBS's "Re-calculate" output is not worth the complexity. Instead we
/// produce **reasonable** values in `[60, max_flush_volume]` that prevent
/// the BambuStudio fallback (8×8=280 default matrix) which causes color
/// bleed-through in real prints.
///
/// Algorithm derived from BambuStudio (AGPL-3.0,
/// `BambuStudio/src/libslic3r/FlushVolCalc.cpp::calc_flush_vol_rgb`).

#include <string_view>

namespace ChromaPrint3D {

constexpr int kMaxFlushVolume = 900; ///< Hard cap (mm³); matches BBS.

/// Stateless calculator: parses hex inputs and computes flush volume.
class FlushVolumeCalculator {
public:
    /// \param min_flush_volume Additive baseline (mm³); BBS-style "extra
    ///        flush" derived from nozzle volume.
    /// \param max_flush_volume Upper clamp (default \ref kMaxFlushVolume).
    FlushVolumeCalculator(int min_flush_volume, int max_flush_volume = kMaxFlushVolume);

    /// Compute the flush volume (mm³) for switching from \p src_hex to \p dst_hex.
    /// Hex strings must be `#RRGGBB` (case-insensitive); on parse failure both
    /// channels are treated as white. Returns the integer volume clamped to
    /// `[0, max_flush_volume]`.
    int Calc(std::string_view src_hex, std::string_view dst_hex) const;

private:
    int min_flush_volume_ = 0;
    int max_flush_volume_ = kMaxFlushVolume;
};

} // namespace ChromaPrint3D
