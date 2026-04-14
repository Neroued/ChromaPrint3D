#include "infrastructure/board_runtime_cache.h"
#include "infrastructure/random_utils.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <system_error>

namespace chromaprint3d::backend {
namespace {

std::optional<SpillableArtifact> WriteBoardModelFile(const std::filesystem::path& temp_dir,
                                                     const std::string& board_id,
                                                     const std::vector<uint8_t>& model_3mf) {
    if (temp_dir.empty() || model_3mf.empty()) return std::nullopt;

    auto file_path = temp_dir / (board_id + "_calibration_board.3mf");
    spdlog::debug("WriteBoardModelFile: board={}, path={}, bytes={}", board_id, file_path.string(),
                  model_3mf.size());
    {
        std::ofstream out(file_path, std::ios::binary | std::ios::trunc);
        if (!out.is_open()) {
            spdlog::error("WriteBoardModelFile: cannot open temp file for board {}: {}", board_id,
                          file_path.string());
            return std::nullopt;
        }
        out.write(reinterpret_cast<const char*>(model_3mf.data()),
                  static_cast<std::streamsize>(model_3mf.size()));
        if (!out.good()) {
            std::error_code ec;
            std::filesystem::remove(file_path, ec);
            spdlog::error("WriteBoardModelFile: write failed for board {} to {}", board_id,
                          file_path.string());
            return std::nullopt;
        }
        out.close();
        if (out.fail()) {
            std::error_code ec;
            std::filesystem::remove(file_path, ec);
            spdlog::error("WriteBoardModelFile: close/flush failed for board {} at {}", board_id,
                          file_path.string());
            return std::nullopt;
        }
    }

    std::error_code ec;
    auto actual_size = std::filesystem::file_size(file_path, ec);
    if (ec || actual_size != model_3mf.size()) {
        spdlog::error("WriteBoardModelFile: size mismatch for board {}: expected={}, actual={}, "
                      "ec={}",
                      board_id, model_3mf.size(), actual_size, ec.message());
        std::filesystem::remove(file_path, ec);
        return std::nullopt;
    }

    spdlog::info("WriteBoardModelFile: board {} written {} bytes to {}", board_id, actual_size,
                 file_path.string());
    return SpillableArtifact::FromFile(std::move(file_path), actual_size);
}

} // namespace

std::size_t BoardGeometryKeyHash::operator()(const BoardGeometryKey& k) const {
    std::size_t h = std::hash<int>{}(k.num_channels);
    h ^= std::hash<int>{}(k.color_layers) + 0x9e3779b9 + (h << 6) + (h >> 2);
    h ^= std::hash<int>{}(k.board_variant) + 0x9e3779b9 + (h << 6) + (h >> 2);
    return h;
}

BoardRuntimeCache::BoardRuntimeCache(std::int64_t ttl_seconds, const std::string& data_dir,
                                     std::int64_t geometry_ttl_seconds,
                                     std::int64_t geometry_max_entries)
    : ttl_seconds_(ttl_seconds), geometry_ttl_seconds_(geometry_ttl_seconds),
      geometry_max_entries_(static_cast<std::size_t>(geometry_max_entries)),
      temp_dir_(std::filesystem::path(data_dir) / "tmp" / "results" / "boards") {
    std::error_code ec;
    std::filesystem::create_directories(temp_dir_, ec);
    if (ec) {
        spdlog::warn("BoardRuntimeCache: failed to create temp dir {}: {}", temp_dir_.string(),
                     ec.message());
        temp_dir_.clear();
        return;
    }
    std::filesystem::permissions(temp_dir_, std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::replace, ec);
    if (ec) {
        spdlog::warn("BoardRuntimeCache: failed to set permissions for {}: {}", temp_dir_.string(),
                     ec.message());
    } else {
        spdlog::info("BoardRuntimeCache: temp dir = {}", temp_dir_.string());
    }
}

std::string BoardRuntimeCache::StoreBoard(ChromaPrint3D::CalibrationBoardResult&& result) {
    std::lock_guard<std::mutex> lock(mtx_);
    CleanupExpiredLocked(std::chrono::steady_clock::now());

    std::string id         = NewBoardId();
    std::size_t model_size = result.model_3mf.size();

    BoardRecord record;
    record.meta       = std::move(result.meta);
    record.created_at = std::chrono::steady_clock::now();

    auto spilled = WriteBoardModelFile(temp_dir_, id, result.model_3mf);
    if (spilled) {
        record.spilled_3mf = std::move(*spilled);
    } else {
        record.model_3mf_fallback = std::move(result.model_3mf);
        if (model_size > 0) {
            spdlog::warn("BoardRuntimeCache: board {} fallback to in-memory 3MF ({} bytes)", id,
                         model_size);
        }
    }

    boards_[id] = std::move(record);
    return id;
}

std::optional<BoardSnapshot> BoardRuntimeCache::FindBoard(const std::string& id) {
    std::lock_guard<std::mutex> lock(mtx_);
    CleanupExpiredLocked(std::chrono::steady_clock::now());
    auto it = boards_.find(id);
    if (it == boards_.end()) return std::nullopt;

    BoardSnapshot snapshot;
    if (it->second.spilled_3mf.has_file()) {
        snapshot.model_3mf_path     = it->second.spilled_3mf.path();
        snapshot.expected_file_size = it->second.spilled_3mf.file_size();
    } else {
        snapshot.model_3mf = it->second.model_3mf_fallback;
    }
    snapshot.meta = it->second.meta;
    return snapshot;
}

std::optional<ChromaPrint3D::CalibrationBoardMeshes>
BoardRuntimeCache::FindGeometry(int num_channels, int color_layers, int board_variant) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto it = geometry_.find(BoardGeometryKey{num_channels, color_layers, board_variant});
    if (it == geometry_.end()) return std::nullopt;
    it->second.last_accessed_at = std::chrono::steady_clock::now();
    return it->second.data;
}

void BoardRuntimeCache::StoreGeometry(int num_channels, int color_layers,
                                      ChromaPrint3D::CalibrationBoardMeshes&& data,
                                      int board_variant) {
    std::lock_guard<std::mutex> lock(mtx_);
    auto now = std::chrono::steady_clock::now();
    CleanupGeometryLocked(now);
    geometry_[BoardGeometryKey{num_channels, color_layers, board_variant}] =
        GeometryRecord{std::move(data), now, now};
}

void BoardRuntimeCache::CleanupExpiredLocked(const std::chrono::steady_clock::time_point& now) {
    for (auto it = boards_.begin(); it != boards_.end();) {
        auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(now - it->second.created_at).count();
        if (elapsed > ttl_seconds_) {
            it = boards_.erase(it);
        } else {
            ++it;
        }
    }
    CleanupGeometryLocked(now);
}

void BoardRuntimeCache::CleanupGeometryLocked(const std::chrono::steady_clock::time_point& now) {
    for (auto it = geometry_.begin(); it != geometry_.end();) {
        auto elapsed =
            std::chrono::duration_cast<std::chrono::seconds>(now - it->second.last_accessed_at)
                .count();
        if (elapsed > geometry_ttl_seconds_) {
            it = geometry_.erase(it);
        } else {
            ++it;
        }
    }
    while (geometry_.size() > geometry_max_entries_) {
        auto oldest =
            std::min_element(geometry_.begin(), geometry_.end(), [](const auto& a, const auto& b) {
                return a.second.last_accessed_at < b.second.last_accessed_at;
            });
        geometry_.erase(oldest);
    }
}

std::string BoardRuntimeCache::NewBoardId() { return detail::RandomHex(16); }

} // namespace chromaprint3d::backend
