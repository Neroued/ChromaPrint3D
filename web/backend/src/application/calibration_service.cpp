#include "application/calibration_service.h"

#include "application/json_serialization.h"
#include "application/request_parsing.h"

#include "infrastructure/task_runtime.h"

#include "chromaprint3d/calib.h"
#include "chromaprint3d/print_profile.h"
#include "chromaprint3d/slicer_preset.h"

#include <spdlog/spdlog.h>

#include <filesystem>
#include <system_error>

namespace chromaprint3d::backend {

using json = nlohmann::json;
using namespace ChromaPrint3D;

CalibrationService::CalibrationService(const ServerConfig& cfg, DataRepository& data,
                                       SessionStore& sessions, BoardRuntimeCache& boards)
    : cfg_(cfg), data_(data), sessions_(sessions), boards_(boards) {}

ServiceResult CalibrationService::GenerateBoard(const json& body) {
    if (!body.contains("palette") || !body["palette"].is_array()) {
        return ServiceResult::Error(400, "invalid_request", "Missing or invalid 'palette' array");
    }
    CalibrationBoardConfig cfg;
    auto& palette_arr       = body["palette"];
    cfg.recipe.num_channels = static_cast<int>(palette_arr.size());
    if (cfg.recipe.num_channels < CalibrationRecipeSpec::kMinChannels ||
        cfg.recipe.num_channels > CalibrationRecipeSpec::kMaxChannels) {
        return ServiceResult::Error(400, "invalid_request", "palette size must be 2-8");
    }
    cfg.palette.resize(palette_arr.size());
    for (size_t i = 0; i < palette_arr.size(); ++i) {
        const auto& ch          = palette_arr[i];
        cfg.palette[i].color    = ch.value("color", "");
        cfg.palette[i].material = ch.value("material", "PLA Basic");
    }
    if (body.contains("color_layers")) cfg.recipe.color_layers = body["color_layers"].get<int>();
    if (body.contains("layer_height_mm")) {
        cfg.layer_height_mm = body["layer_height_mm"].get<float>();
    }

    std::string raw_nozzle = body.value("nozzle_size", "n04");
    std::string raw_face   = body.value("face_orientation", "faceup");
    NozzleSize nozzle      = ParseNozzleSize(raw_nozzle);
    FaceOrientation face   = ParseFaceOrientation(raw_face);
    spdlog::info("GenerateBoard: nozzle_size='{}' -> {}, face_orientation='{}' -> {}", raw_nozzle,
                 NozzleSizeTag(nozzle), raw_face, FaceOrientationTag(face));

    auto presets_path = std::filesystem::path(cfg_.data_dir) / "presets";
    bool has_preset   = std::filesystem::is_directory(presets_path);

    try {
        const int num_ch   = cfg.recipe.num_channels;
        const int c_layers = cfg.recipe.color_layers;

        CalibrationBoardResult result;
        auto cached = boards_.FindGeometry(num_ch, c_layers);

        if (has_preset) {
            FilamentConfig fil_config = FilamentConfig::LoadFromDir(presets_path.string());
            PrintProfile profile;
            profile.layer_height_mm  = cfg.layer_height_mm;
            profile.palette          = cfg.palette;
            profile.nozzle_size      = nozzle;
            profile.face_orientation = face;
            for (size_t i = 0; i < profile.palette.size(); ++i) {
                auto& ch = profile.palette[i];
                if (ch.hex_color.empty()) {
                    ch.hex_color = fil_config.ResolveHexColor(ch.color, static_cast<int>(i));
                }
            }
            cfg.palette = profile.palette;
            auto preset = SlicerPreset::FromProfile(presets_path.string(), profile, &fil_config);

            if (cached) {
                result = BuildResultFromMeshes(*cached, cfg.palette, preset, face);
            } else {
                auto meshes = GenCalibrationBoardMeshes(cfg);
                result      = BuildResultFromMeshes(meshes, cfg.palette, preset, face);
                boards_.StoreGeometry(num_ch, c_layers, std::move(meshes));
            }
        } else {
            if (cached) {
                result = BuildResultFromMeshes(*cached, cfg.palette, face);
            } else {
                auto meshes = GenCalibrationBoardMeshes(cfg);
                result      = BuildResultFromMeshes(meshes, cfg.palette, face);
                boards_.StoreGeometry(num_ch, c_layers, std::move(meshes));
            }
        }

        std::string meta_str = result.meta.ToJsonString();
        std::string board_id = boards_.StoreBoard(std::move(result));
        return ServiceResult::Success(200,
                                      {{"board_id", board_id}, {"meta", json::parse(meta_str)}});
    } catch (const std::exception& e) {
        return ServiceResult::Error(500, "generate_failed", e.what());
    }
}

ServiceResult CalibrationService::Generate8ColorBoard(const json& body) {
    if (!data_.RecipeStore().loaded) {
        return ServiceResult::Error(503, "service_unavailable",
                                    "8-color recipe data not available on server");
    }
    if (!body.contains("palette") || !body["palette"].is_array()) {
        return ServiceResult::Error(400, "invalid_request", "Missing or invalid 'palette' array");
    }
    if (!body.contains("board_index") || !body["board_index"].is_number_integer()) {
        return ServiceResult::Error(400, "invalid_request", "Missing or invalid 'board_index'");
    }

    auto& palette_arr = body["palette"];
    const int num_ch  = static_cast<int>(palette_arr.size());
    if (num_ch != data_.RecipeStore().num_channels) {
        return ServiceResult::Error(400, "invalid_request",
                                    "palette size must be " +
                                        std::to_string(data_.RecipeStore().num_channels));
    }

    const int board_index = body["board_index"].get<int>();
    const auto* board_set = data_.RecipeStore().FindBoard(board_index);
    if (!board_set) {
        return ServiceResult::Error(400, "invalid_request",
                                    "Invalid board_index: " + std::to_string(board_index));
    }

    CalibrationBoardConfig cfg;
    cfg.recipe.num_channels     = num_ch;
    cfg.recipe.color_layers     = data_.RecipeStore().color_layers;
    cfg.recipe.layer_order      = (data_.RecipeStore().layer_order == "Bottom2Top")
                                      ? LayerOrder::Bottom2Top
                                      : LayerOrder::Top2Bottom;
    cfg.base_layers             = data_.RecipeStore().base_layers;
    cfg.base_channel_idx        = data_.RecipeStore().base_channel_idx;
    cfg.layer_height_mm         = data_.RecipeStore().layer_height_mm;
    cfg.layout.line_width_mm    = data_.RecipeStore().layout.line_width_mm;
    cfg.layout.resolution_scale = data_.RecipeStore().layout.resolution_scale;
    cfg.layout.tile_factor      = data_.RecipeStore().layout.tile_factor;
    cfg.layout.gap_factor       = data_.RecipeStore().layout.gap_factor;
    cfg.layout.margin_factor    = data_.RecipeStore().layout.margin_factor;
    cfg.palette.resize(palette_arr.size());
    for (size_t i = 0; i < palette_arr.size(); ++i) {
        const auto& ch          = palette_arr[i];
        cfg.palette[i].color    = ch.value("color", "");
        cfg.palette[i].material = ch.value("material", "PLA Basic");
    }

    std::string raw_nozzle = body.value("nozzle_size", "n04");
    std::string raw_face   = body.value("face_orientation", "faceup");
    NozzleSize nozzle      = ParseNozzleSize(raw_nozzle);
    FaceOrientation face   = ParseFaceOrientation(raw_face);
    spdlog::info("Generate8ColorBoard: nozzle_size='{}' -> {}, face_orientation='{}' -> {}",
                 raw_nozzle, NozzleSizeTag(nozzle), raw_face, FaceOrientationTag(face));

    auto presets_path = std::filesystem::path(cfg_.data_dir) / "presets";
    bool has_preset   = std::filesystem::is_directory(presets_path);

    try {
        CalibrationBoardResult result;
        auto cached = boards_.FindGeometry(num_ch, cfg.recipe.color_layers, board_index);

        auto build_with_preset = [&](const CalibrationBoardMeshes& meshes_ref) {
            if (has_preset) {
                FilamentConfig fil_config = FilamentConfig::LoadFromDir(presets_path.string());
                PrintProfile profile;
                profile.layer_height_mm  = cfg.layer_height_mm;
                profile.palette          = cfg.palette;
                profile.nozzle_size      = nozzle;
                profile.face_orientation = face;
                for (size_t i = 0; i < profile.palette.size(); ++i) {
                    auto& ch = profile.palette[i];
                    if (ch.hex_color.empty()) {
                        ch.hex_color = fil_config.ResolveHexColor(ch.color, static_cast<int>(i));
                    }
                }
                cfg.palette = profile.palette;
                auto preset =
                    SlicerPreset::FromProfile(presets_path.string(), profile, &fil_config);
                return BuildResultFromMeshes(meshes_ref, cfg.palette, preset, face);
            }
            return BuildResultFromMeshes(meshes_ref, cfg.palette, face);
        };

        if (cached) {
            result = build_with_preset(*cached);
        } else {
            auto meta = BuildCalibrationBoardMetaCustom(cfg, board_set->grid_rows,
                                                        board_set->grid_cols, board_set->recipes);
            meta.name += "_board" + std::to_string(board_index);
            auto meshes = GenCalibrationBoardMeshesFromMeta(std::move(meta));
            result      = build_with_preset(meshes);
            boards_.StoreGeometry(num_ch, cfg.recipe.color_layers, std::move(meshes), board_index);
        }

        std::string meta_str = result.meta.ToJsonString();
        std::string board_id = boards_.StoreBoard(std::move(result));
        return ServiceResult::Success(200,
                                      {{"board_id", board_id}, {"meta", json::parse(meta_str)}});
    } catch (const std::exception& e) {
        return ServiceResult::Error(500, "generate_failed", e.what());
    }
}

ServiceResult CalibrationService::DownloadBoardModel(const std::string& board_id,
                                                     TaskArtifact& out) {
    auto board = boards_.FindBoard(board_id);
    if (!board) return ServiceResult::Error(404, "not_found", "Board not found or expired");
    std::string filename = board->meta.name.empty() ? "calibration_board" : board->meta.name;
    filename += ".3mf";

    constexpr const char* kModelContentType =
        "application/vnd.ms-package.3dmanufacturing-3dmodel+xml";
    if (board->has_file_backed_model()) {
        const auto& fpath = board->model_3mf_path;
        std::error_code ec;
        bool exists      = std::filesystem::exists(fpath, ec);
        auto actual_size = exists ? std::filesystem::file_size(fpath, ec) : 0ULL;
        spdlog::debug("DownloadBoardModel: board={}, path={}, exists={}, actual_size={}, "
                      "expected_size={}",
                      board_id, fpath.string(), exists, actual_size, board->expected_file_size);
        if (!exists) {
            spdlog::error("DownloadBoardModel: file missing for board {}: {}", board_id,
                          fpath.string());
            return ServiceResult::Error(404, "not_found", "Board model file missing from disk");
        }
        if (board->expected_file_size > 0 && actual_size != board->expected_file_size) {
            spdlog::error("DownloadBoardModel: file size mismatch for board {}: expected={}, "
                          "actual={}",
                          board_id, board->expected_file_size, actual_size);
            return ServiceResult::Error(500, "corrupt_file",
                                        "Board model file is incomplete or corrupt");
        }
        out = TaskArtifact{{}, fpath, kModelContentType, std::move(filename)};
        return ServiceResult::Success(200, json::object());
    }
    if (board->model_3mf.empty()) {
        return ServiceResult::Error(404, "not_found", "Board model not available");
    }
    out = TaskArtifact{std::move(board->model_3mf), {}, kModelContentType, std::move(filename)};
    return ServiceResult::Success(200, json::object());
}

ServiceResult CalibrationService::DownloadBoardMeta(const std::string& board_id,
                                                    TaskArtifact& out) {
    auto board = boards_.FindBoard(board_id);
    if (!board) return ServiceResult::Error(404, "not_found", "Board not found or expired");
    std::string filename =
        board->meta.name.empty() ? "calibration_board_meta" : board->meta.name + "_meta";
    filename += ".json";
    auto meta_json = board->meta.ToJsonString();
    std::vector<uint8_t> bytes(meta_json.begin(), meta_json.end());
    TaskArtifact artifact;
    artifact.bytes        = std::move(bytes);
    artifact.content_type = "application/json";
    artifact.filename     = std::move(filename);
    out                   = std::move(artifact);
    return ServiceResult::Success(200, json::object());
}

ServiceResult CalibrationService::LocateBoard(const std::vector<uint8_t>& image,
                                              const std::string& meta_json) {
    auto valid = ValidateDecodedImage(image, cfg_.max_pixels_per_image);
    if (!valid.ok) return valid;

    CalibrationBoardMeta meta;
    try {
        meta = CalibrationBoardMeta::FromJsonString(meta_json);
    } catch (const std::exception& e) {
        return ServiceResult::Error(400, "invalid_meta",
                                    "Invalid meta JSON: " + std::string(e.what()));
    }

    try {
        auto result      = LocateCalibrationBoard(image, meta);
        json corners_arr = json::array();
        for (const auto& c : result.corners) { corners_arr.push_back(json::array({c[0], c[1]})); }
        json out            = json::object();
        out["corners"]      = std::move(corners_arr);
        out["image_width"]  = result.image_width;
        out["image_height"] = result.image_height;
        return ServiceResult::Success(200, std::move(out));
    } catch (const std::exception& e) {
        return ServiceResult::Error(422, "locate_failed", e.what());
    }
}

ServiceResult CalibrationService::BuildColorDb(const std::string& owner,
                                               const std::vector<uint8_t>& image,
                                               const std::string& meta_json,
                                               const std::string& db_name,
                                               const std::string& corners_json) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto valid = ValidateDecodedImage(image, cfg_.max_pixels_per_image);
    if (!valid.ok) return valid;
    if (!SessionStore::IsValidDbName(db_name)) {
        return ServiceResult::Error(400, "invalid_name",
                                    "Invalid ColorDB name ([a-zA-Z0-9_], length 1-64)");
    }
    if (data_.ColorDbCache().FindByName(db_name)) {
        return ServiceResult::Error(409, "name_conflict", "Name conflicts with global database");
    }

    CalibrationBoardMeta meta;
    try {
        meta = CalibrationBoardMeta::FromJsonString(meta_json);
    } catch (const std::exception& e) {
        return ServiceResult::Error(400, "invalid_meta",
                                    "Invalid meta JSON: " + std::string(e.what()));
    }

    std::optional<std::array<std::array<float, 2>, 4>> corners_opt;
    if (!corners_json.empty()) {
        try {
            auto j = json::parse(corners_json);
            if (!j.is_array() || j.size() != 4) {
                return ServiceResult::Error(400, "invalid_corners",
                                            "corners must be an array of 4 [x,y] pairs");
            }
            std::array<std::array<float, 2>, 4> pts;
            for (size_t i = 0; i < 4; ++i) {
                if (!j[i].is_array() || j[i].size() != 2) {
                    return ServiceResult::Error(400, "invalid_corners",
                                                "Each corner must be [x, y]");
                }
                pts[i] = {j[i][0].get<float>(), j[i][1].get<float>()};
            }
            corners_opt = pts;
        } catch (const json::exception& e) {
            return ServiceResult::Error(400, "invalid_corners",
                                        "Failed to parse corners: " + std::string(e.what()));
        }
    }

    try {
        ColorDB db = corners_opt ? GenColorDBFromBuffer(image, meta, *corners_opt)
                                 : GenColorDBFromBuffer(image, meta);
        db.name    = db_name;
        std::string err;
        if (!sessions_.PutSessionDb(owner, db, &err)) {
            return ServiceResult::Error(429, "session_limit", err);
        }
        auto out      = ColorDbBuildResultToJson(db);
        out["source"] = "session";
        return ServiceResult::Success(200, std::move(out));
    } catch (const std::exception& e) {
        return ServiceResult::Error(500, "build_failed", e.what());
    }
}

} // namespace chromaprint3d::backend
