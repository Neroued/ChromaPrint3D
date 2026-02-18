#pragma once

/// \file print_profile.h
/// \brief Print mode enumeration and print profile configuration.

#include "common.h"
#include "color_db.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ChromaPrint3D {

/// Supported multi-color print modes.
enum class PrintMode : uint8_t {
    Mode0p08x5  = 0, ///< 0.08 mm layer height, 5 color layers.
    Mode0p04x10 = 1, ///< 0.04 mm layer height, 10 color layers.
};

/// Describes a complete printing configuration derived from one or more ColorDBs.
struct PrintProfile {
    PrintMode mode = PrintMode::Mode0p08x5;

    float max_color_thickness_mm = 0.4f;
    float layer_height_mm        = 0.08f;
    int color_layers             = 5;

    float line_width_mm = 0.42f;

    int base_layers      = 0;
    int base_channel_idx = 0;

    LayerOrder layer_order = LayerOrder::Top2Bottom;
    std::vector<Channel> palette;

    size_t NumChannels() const { return palette.size(); }

    /// Validate internal consistency; throws on failure.
    void Validate() const;

    /// Convert this profile into a ColorDB (no entries, metadata only).
    ColorDB ToColorDB(const std::string& name = "PrintProfileDB") const;

    /// Merge multiple ColorDBs into a unified print profile for the given mode.
    static PrintProfile BuildFromColorDBs(std::span<const ColorDB> dbs, PrintMode mode);

    /// Keep only channels whose normalized key is in \p allowed_keys.
    /// The base channel is always retained.
    void FilterChannels(const std::vector<std::string>& allowed_keys);
};

} // namespace ChromaPrint3D
