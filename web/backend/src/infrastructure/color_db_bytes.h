#pragma once

#include "chromaprint3d/color_db.h"

#include <cstddef>
#include <memory>
#include <unordered_map>

namespace chromaprint3d::backend::detail {

inline std::size_t EstimateColorDbBytes(const ChromaPrint3D::ColorDB& db) {
    std::size_t bytes = 0;
    for (const auto& entry : db.entries) {
        bytes += sizeof(ChromaPrint3D::Entry) + entry.recipe.capacity();
    }
    for (const auto& ch : db.palette) {
        bytes += ch.color.capacity() + ch.material.capacity() + ch.hex_color.capacity();
    }
    return bytes;
}

inline std::size_t EstimateColorDbMapBytes(
    const std::unordered_map<std::string, std::shared_ptr<const ChromaPrint3D::ColorDB>>&
        color_dbs) {
    std::size_t bytes = 0;
    for (const auto& [_, db] : color_dbs) {
        if (db) { bytes += EstimateColorDbBytes(*db); }
    }
    return bytes;
}

} // namespace chromaprint3d::backend::detail
