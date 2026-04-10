#pragma once

/// \file print_profile.h
/// \brief Print profile configuration.

#include "common.h"
#include "color_db.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace ChromaPrint3D {

struct FilamentConfig;

/// Describes a complete printing configuration derived from one or more ColorDBs.
struct PrintProfile {
    static constexpr int kDefaultColorLayers = 5;
    static constexpr int kSoftMaxColorLayers = 10;

    float layer_height_mm = 0.08f;
    int color_layers      = kDefaultColorLayers;

    float line_width_mm = 0.42f;

    int base_layers      = 0;
    int base_channel_idx = 0;

    LayerOrder layer_order = LayerOrder::Top2Bottom;

    NozzleSize nozzle_size           = NozzleSize::N04;
    FaceOrientation face_orientation = FaceOrientation::FaceUp;

    std::vector<Channel> palette;

    size_t NumChannels() const { return palette.size(); }

    /// Validate internal consistency; throws on failure.
    void Validate() const;

    /// Convert this profile into a ColorDB (no entries, metadata only).
    ColorDB ToColorDB(const std::string& name = "PrintProfileDB") const;

    /// Merge multiple ColorDBs into a unified print profile.
    /// \p color_layers is the user-selected quality parameter (default 5).
    /// \p layer_height_mm controls the physical layer height (0 = derive from first DB).
    /// If \p config is non-null, uses its color table for hex resolution.
    static PrintProfile BuildFromColorDBs(std::span<const ColorDB> dbs, int color_layers,
                                          float layer_height_mm        = 0.0f,
                                          const FilamentConfig* config = nullptr);

    /// Keep only channels whose normalized key is in \p allowed_keys.
    /// The base channel is always retained.
    void FilterChannels(const std::vector<std::string>& allowed_keys);
};

} // namespace ChromaPrint3D
