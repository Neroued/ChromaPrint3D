#pragma once

#include "session.h"

#include "chromaprint3d/calib.h"

#include <chrono>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

using namespace ChromaPrint3D;

struct BoardCacheEntry {
    std::vector<uint8_t> model_3mf;
    CalibrationBoardMeta meta;
    std::chrono::steady_clock::time_point created_at;
};

struct BoardCache {
    std::mutex mtx;
    std::unordered_map<std::string, BoardCacheEntry> entries;
    int ttl_seconds = 600;

    std::string Store(CalibrationBoardResult&& result) {
        std::lock_guard<std::mutex> lock(mtx);
        CleanExpiredLocked();
        std::string id = GenerateUUID();
        BoardCacheEntry entry;
        entry.model_3mf  = std::move(result.model_3mf);
        entry.meta       = std::move(result.meta);
        entry.created_at = std::chrono::steady_clock::now();
        entries[id]      = std::move(entry);
        return id;
    }

    const BoardCacheEntry* Find(const std::string& id) {
        std::lock_guard<std::mutex> lock(mtx);
        auto it = entries.find(id);
        if (it == entries.end()) { return nullptr; }
        return &it->second;
    }

    void CleanExpiredLocked() {
        auto now = std::chrono::steady_clock::now();
        for (auto it = entries.begin(); it != entries.end();) {
            auto elapsed =
                std::chrono::duration_cast<std::chrono::seconds>(now - it->second.created_at)
                    .count();
            if (elapsed > ttl_seconds) {
                it = entries.erase(it);
            } else {
                ++it;
            }
        }
    }
};
