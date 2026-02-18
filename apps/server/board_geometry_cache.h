#pragma once

#include "chromaprint3d/calib.h"

#include <spdlog/spdlog.h>

#include <mutex>
#include <unordered_map>

using namespace ChromaPrint3D;

struct BoardGeometryKey {
    int num_channels = 0;
    int color_layers = 0;

    bool operator==(const BoardGeometryKey& other) const {
        return num_channels == other.num_channels && color_layers == other.color_layers;
    }
};

struct BoardGeometryKeyHash {
    std::size_t operator()(const BoardGeometryKey& k) const {
        std::size_t h = std::hash<int>{}(k.num_channels);
        h ^= std::hash<int>{}(k.color_layers) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

struct BoardGeometryCache {
    std::mutex mtx;
    std::unordered_map<BoardGeometryKey, CalibrationBoardMeshes, BoardGeometryKeyHash> entries;

    /// Look up cached meshes. Returns nullptr if not found.
    const CalibrationBoardMeshes* Find(int num_channels, int color_layers) {
        std::lock_guard<std::mutex> lock(mtx);
        BoardGeometryKey key{num_channels, color_layers};
        auto it = entries.find(key);
        if (it == entries.end()) { return nullptr; }
        return &it->second;
    }

    /// Store meshes in the cache.
    void Store(int num_channels, int color_layers, CalibrationBoardMeshes&& data) {
        std::lock_guard<std::mutex> lock(mtx);
        BoardGeometryKey key{num_channels, color_layers};
        entries[key] = std::move(data);
        spdlog::info("BoardGeometryCache: stored entry for {}ch/{}L ({} meshes)",
                     num_channels, color_layers, entries[key].meshes.size());
    }

    /// Number of cached entries.
    std::size_t Size() {
        std::lock_guard<std::mutex> lock(mtx);
        return entries.size();
    }
};
