#pragma once

// Internal recipe conversion utilities.
// NOT part of the public API.

#include "chromaprint3d/color_db.h"
#include "chromaprint3d/print_profile.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace ChromaPrint3D {
namespace detail {

struct PreparedDB {
    const ColorDB* db = nullptr;
    std::vector<int> source_to_target_channel;
    std::unique_ptr<ColorDB> filtered_db;
};

bool ConvertRecipeToProfile(const Entry& entry, const PreparedDB& prepared_db,
                            const PrintProfile& profile, std::vector<uint8_t>& out_recipe);

std::vector<PreparedDB> PrepareDBs(std::span<const ColorDB> dbs, const PrintProfile& profile);

} // namespace detail
} // namespace ChromaPrint3D
