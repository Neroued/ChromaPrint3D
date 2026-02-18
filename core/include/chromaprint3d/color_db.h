/// \file color_db.h
/// \brief Color database for color matching and recipe lookup.

#pragma once

#include "common.h"
#include "color.h"
#include "kdtree.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ChromaPrint3D {

/// Color database entry containing a color and its printing recipe.
struct Entry {
    Lab lab; ///< Target color in Lab color space.
    std::vector<uint8_t> recipe; ///< Printing recipe (layer thicknesses per channel, size == color_layers).

    /// Returns the number of color layers in the recipe.
    /// \return Number of color layers
    size_t ColorLayers() const { return recipe.size(); }
};

/// Channel configuration (color and material name).
struct Channel {
    std::string color    = "Default Color";    ///< Color name (e.g., "Cyan").
    std::string material = "Default Material"; ///< Material name (e.g., "PLA Basic").
};

/// Color database for color matching and recipe lookup.
class ColorDB {
public:
    std::string name = "Default ColorDB"; ///< Database name.

    int max_color_layers = 0; ///< Maximum number of color layers in recipes.

    int base_layers      = 0; ///< Number of base layers.
    int base_channel_idx = 0; ///< Channel index used for base layers.

    float layer_height_mm = 0.08f; ///< Layer height in millimeters.
    float line_width_mm   = 0.42f; ///< Line width in millimeters.

    LayerOrder layer_order = LayerOrder::Top2Bottom; ///< Printing order for layers.

    std::vector<Channel> palette; ///< Channel palette (size == num_channels).
    std::vector<Entry> entries; ///< Color entries in the database.

    ColorDB()  = default;
    ~ColorDB() = default;
    ColorDB(const ColorDB& other);
    ColorDB(ColorDB&& other) noexcept;
    ColorDB& operator=(const ColorDB& other);
    ColorDB& operator=(ColorDB&& other) noexcept;

    /// Returns the number of channels in the palette.
    /// \return Number of channels
    size_t NumChannels() const { return palette.size(); }

    /// Loads a ColorDB from a JSON file.
    /// \param path Path to the JSON file
    /// \return Loaded ColorDB
    static ColorDB LoadFromJson(const std::string& path);
    /// Creates a ColorDB from a JSON string.
    /// \param json_str JSON string representation
    /// \return Parsed ColorDB
    static ColorDB FromJsonString(const std::string& json_str);

    /// Saves this ColorDB to a JSON file.
    /// \param path Path to save the JSON file
    void SaveToJson(const std::string& path) const;
    /// Converts this ColorDB to a JSON string.
    /// \return JSON string representation
    std::string ToJsonString() const;

    /// Finds the nearest color entry to the target Lab color.
    /// \param target Target Lab color
    /// \return Reference to the nearest entry
    const Entry& NearestEntry(const Lab& target) const;

    /// Finds the nearest color entry to the target RGB color.
    /// \param target Target RGB color
    /// \return Reference to the nearest entry
    const Entry& NearestEntry(const Rgb& target) const;

    /// Finds the k nearest color entries to the target Lab color.
    /// \param target Target Lab color
    /// \param k Number of nearest entries to find
    /// \return Vector of pointers to the k nearest entries (sorted by distance)
    std::vector<const Entry*> NearestEntries(const Lab& target, std::size_t k) const;

    /// Finds the k nearest color entries to the target RGB color.
    /// \param target Target RGB color
    /// \param k Number of nearest entries to find
    /// \return Vector of pointers to the k nearest entries (sorted by distance)
    std::vector<const Entry*> NearestEntries(const Rgb& target, std::size_t k) const;

private:
    struct LabProj {
        const Lab& operator()(const Entry& entry) const { return entry.lab; }
    };

    struct RgbProj {
        Rgb operator()(const Entry& entry) const { return entry.lab.ToRgb(); }
    };

    using KdIndex = std::size_t;
    using LabTree = kdt::KDTree<Entry, 3, LabProj, KdIndex, float>;
    using RgbTree = kdt::KDTree<Entry, 3, RgbProj, KdIndex, float>;

    void BuildKDTree() const;
    void EnsureKDTree() const;
    void ResetKDTreeCache() const;

    mutable std::vector<KdIndex> kd_indices_;
    mutable LabTree lab_tree_;
    mutable RgbTree rgb_tree_;
    mutable std::size_t kd_entries_size_ = 0;
};

} // namespace ChromaPrint3D