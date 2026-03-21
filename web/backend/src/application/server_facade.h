#pragma once

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

struct ServiceResult {
    bool ok         = false;
    int status_code = 500;
    std::string code;
    std::string message;
    nlohmann::json data = nlohmann::json::object();

    static ServiceResult Success(int status, nlohmann::json d);
    static ServiceResult Error(int status, std::string c, std::string m);
};

class ServerFacade {
public:
    explicit ServerFacade(ServerConfig cfg);

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

    ServiceResult RecipeEditorSummary(const std::string& owner, const std::string& task_id) const;
    ServiceResult RecipeEditorAlternatives(const std::string& owner, const std::string& task_id,
                                           const std::string& body_json);
    ServiceResult RecipeEditorReplace(const std::string& owner, const std::string& task_id,
                                      const std::string& body_json);
    ServiceResult RecipeEditorGenerate(const std::string& owner, const std::string& task_id);

    ServiceResult ListTasks(const std::string& owner) const;
    ServiceResult GetTask(const std::string& owner, const std::string& id) const;
    ServiceResult DeleteTask(const std::string& owner, const std::string& id);
    ServiceResult DownloadTaskArtifact(const std::string& owner, const std::string& id,
                                       const std::string& artifact, TaskArtifact& out) const;

    ServiceResult MattingMethods() const;

    ServiceResult GenerateBoard(const nlohmann::json& body);
    ServiceResult Generate8ColorBoard(const nlohmann::json& body);
    ServiceResult DownloadBoardModel(const std::string& board_id, TaskArtifact& out);
    ServiceResult DownloadBoardMeta(const std::string& board_id, TaskArtifact& out);
    ServiceResult LocateBoard(const std::vector<uint8_t>& image, const std::string& meta_json);
    ServiceResult BuildColorDb(const std::string& owner, const std::vector<uint8_t>& image,
                               const std::string& meta_json, const std::string& db_name,
                               const std::string& corners_json = "");

private:
    static nlohmann::json ColorDbInfoToJson(const ChromaPrint3D::ColorDB& db,
                                            const std::string& material_type = "",
                                            const std::string& vendor        = "");
    static nlohmann::json ColorDbBuildResultToJson(const ChromaPrint3D::ColorDB& db);
    static nlohmann::json TaskToJson(const TaskSnapshot& task);
    static const char* ConvertStageToString(ChromaPrint3D::ConvertStage stage);

    ServiceResult ParseJsonObject(const std::optional<std::string>& raw, nlohmann::json& out) const;
    ServiceResult ValidateDecodedImage(const std::vector<uint8_t>& image) const;
    ServiceResult ResolveSelectedColorDbs(
        const nlohmann::json& params, const std::optional<SessionSnapshot>& session,
        std::vector<const ChromaPrint3D::ColorDB*>& out_dbs,
        std::vector<std::shared_ptr<const ChromaPrint3D::ColorDB>>& session_owned,
        bool& has_bambu_pla) const;

    ServiceResult BuildRasterRequest(const nlohmann::json& params,
                                     const std::vector<uint8_t>& image,
                                     const std::string& image_name,
                                     const std::optional<SessionSnapshot>& session,
                                     ChromaPrint3D::ConvertRasterRequest& out) const;
    ServiceResult BuildVectorRequest(const nlohmann::json& params, const std::vector<uint8_t>& svg,
                                     const std::string& svg_name,
                                     const std::optional<SessionSnapshot>& session,
                                     ChromaPrint3D::ConvertVectorRequest& out) const;
    ServiceResult BuildVectorizeConfig(const nlohmann::json& params,
                                       neroued::vectorizer::VectorizerConfig& out) const;

    ServerConfig cfg_;
    DataRepository data_;
    SessionStore sessions_;
    TaskRuntime tasks_;
    BoardRuntimeCache boards_;
};

} // namespace chromaprint3d::backend
