#pragma once

#include "application/announcement_service.h"
#include "application/calibration_service.h"
#include "application/color_db_service.h"
#include "application/convert_service.h"
#include "application/matting_vectorize_service.h"
#include "application/recipe_editor_service.h"
#include "application/service_result.h"
#include "application/task_service.h"
#include "config/server_config.h"
#include "infrastructure/board_runtime_cache.h"
#include "infrastructure/data_repository.h"
#include "infrastructure/session_store.h"
#include "infrastructure/task_runtime.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace chromaprint3d::backend {

class ServerFacade {
public:
    explicit ServerFacade(ServerConfig cfg);

    ServerFacade(const ServerFacade&)            = delete;
    ServerFacade& operator=(const ServerFacade&) = delete;
    ServerFacade(ServerFacade&&)                 = delete;
    ServerFacade& operator=(ServerFacade&&)      = delete;

    const ServerConfig& Config() const { return cfg_; }

    bool IsCrossOriginMode() const { return !cfg_.cors_origin.empty(); }

    std::string EnsureSession(const std::optional<std::string>& existing_token, bool* created);
    std::optional<SessionSnapshot> SessionSnapshotByToken(const std::string& token);

    ServiceResult Health() const;
    ServiceResult ConvertDefaults() const;
    ServiceResult VectorizeDefaults() const;
    ServiceResult ListDatabases(const std::optional<std::string>& session_token);
    ServiceResult ListSessionDatabases(const std::string& token);
    ServiceResult UploadSessionDatabase(const std::string& token, const std::string& json_content,
                                        const std::optional<std::string>& override_name);
    ServiceResult DeleteSessionDatabase(const std::string& token, const std::string& name);
    ServiceResult DownloadSessionDatabase(const std::string& token, const std::string& name,
                                          std::string& json_out, std::string& filename_out);

    ServiceResult SubmitConvertRaster(const std::string& owner, const std::vector<uint8_t>& image,
                                      const std::string& image_name,
                                      const std::optional<std::string>& params_json);
    ServiceResult SubmitConvertVector(const std::string& owner, const std::vector<uint8_t>& svg,
                                      const std::string& svg_name,
                                      const std::optional<std::string>& params_json);
    ServiceResult AnalyzeVectorWidth(const std::vector<uint8_t>& svg,
                                     const std::optional<std::string>& params_json);
    ServiceResult SubmitMatting(const std::string& owner, const std::vector<uint8_t>& image,
                                const std::string& image_name, const std::string& method);
    ServiceResult PostprocessMatting(const std::string& owner, const std::string& task_id,
                                     const std::string& body_json);
    ServiceResult SubmitVectorize(const std::string& owner, const std::vector<uint8_t>& image,
                                  const std::string& image_name,
                                  const std::optional<std::string>& params_json);

    ServiceResult SubmitConvertRasterMatchOnly(const std::string& owner,
                                               const std::vector<uint8_t>& image,
                                               const std::string& image_name,
                                               const std::optional<std::string>& params_json);
    ServiceResult SubmitConvertVectorMatchOnly(const std::string& owner,
                                               const std::vector<uint8_t>& svg,
                                               const std::string& svg_name,
                                               const std::optional<std::string>& params_json);

    ServiceResult RecipeEditorSummary(const std::string& owner, const std::string& task_id);
    ServiceResult RecipeEditorAlternatives(const std::string& owner, const std::string& task_id,
                                           const std::string& body_json);
    ServiceResult RecipeEditorReplace(const std::string& owner, const std::string& task_id,
                                      const std::string& body_json);
    ServiceResult RecipeEditorGenerate(const std::string& owner, const std::string& task_id);
    ServiceResult RecipeEditorPredict(const std::string& owner, const std::string& task_id,
                                      const std::string& body_json);

    ServiceResult ListTasks(const std::string& owner) const;
    ServiceResult GetTask(const std::string& owner, const std::string& id);
    ServiceResult DeleteTask(const std::string& owner, const std::string& id);
    ServiceResult DownloadTaskArtifact(const std::string& owner, const std::string& id,
                                       const std::string& artifact, TaskArtifact& out);

    ServiceResult MattingMethods() const;
    nlohmann::json GetModelPackInfo() const;

    ServiceResult GenerateBoard(const nlohmann::json& body);
    ServiceResult Generate8ColorBoard(const nlohmann::json& body);
    ServiceResult DownloadBoardModel(const std::string& board_id, TaskArtifact& out);
    ServiceResult DownloadBoardMeta(const std::string& board_id, TaskArtifact& out);
    ServiceResult LocateBoard(const std::vector<uint8_t>& image, const std::string& meta_json);
    ServiceResult BuildColorDb(const std::string& owner, const std::vector<uint8_t>& image,
                               const std::string& meta_json, const std::string& db_name,
                               const std::string& corners_json = "");

    // Announcements (public read, token-guarded writes).
    ServiceResult ListAnnouncements() const;
    ServiceResult UpsertAnnouncement(const nlohmann::json& body);
    ServiceResult DeleteAnnouncement(const std::string& id);

private:
    ServerConfig cfg_;
    DataRepository data_;
    SessionStore sessions_;
    TaskRuntime tasks_;
    BoardRuntimeCache boards_;

    ColorDbService color_db_svc_;
    TaskService task_svc_;
    ConvertService convert_svc_;
    MattingVectorizeService matting_svc_;
    RecipeEditorService recipe_svc_;
    CalibrationService calib_svc_;
    AnnouncementService announcement_svc_;
};

} // namespace chromaprint3d::backend
