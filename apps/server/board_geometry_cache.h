#pragma once

#include "chromaprint3d/calib.h"

#include <spdlog/spdlog.h>

#include <mutex>
#include <unordered_map>

using namespace ChromaPrint3D;

struct BoardGeometryKey {
    int num_channels  = 0;
    int color_layers  = 0;
    int board_variant = 0; // 0 = standard (all recipes), 1/2 = 8-color board index

    bool operator==(const BoardGeometryKey& other) const {
        return num_channels == other.num_channels && color_layers == other.color_layers &&
               board_variant == other.board_variant;
    }
};

struct BoardGeometryKeyHash {
    std::size_t operator()(const BoardGeometryKey& k) const {
        std::size_t h = std::hash<int>{}(k.num_channels);
        h ^= std::hash<int>{}(k.color_layers) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>{}(k.board_variant) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct BoardGeometryCache {
    std::mutex mtx;
    std::unordered_map<BoardGeometryKey, CalibrationBoardMeshes, BoardGeometryKeyHash> entries;

    const CalibrationBoardMeshes* Find(int num_channels, int color_layers,
                                       int board_variant = 0) {
        std::lock_guard<std::mutex> lock(mtx);
        BoardGeometryKey key{num_channels, color_layers, board_variant};
        auto it = entries.find(key);
        if (it == entries.end()) { return nullptr; }
        return &it->second;
    }

    void Store(int num_channels, int color_layers, CalibrationBoardMeshes&& data,
               int board_variant = 0) {
        std::lock_guard<std::mutex> lock(mtx);
        BoardGeometryKey key{num_channels, color_layers, board_variant};
        entries[key] = std::move(data);
        spdlog::info("BoardGeometryCache: stored entry for {}ch/{}L/v{} ({} meshes)",
                     num_channels, color_layers, board_variant, entries[key].meshes.size());
    }

    std::size_t Size() {
        std::lock_guard<std::mutex> lock(mtx);
        return entries.size();
    }
};
