#pragma once

#include "application/server_facade.h"

#include <drogon/drogon.h>

#include <functional>
#include <optional>
#include <string>

namespace chromaprint3d::backend {

class ApiV1Controller : public drogon::HttpController<ApiV1Controller> {
public:
    METHOD_LIST_BEGIN
    ADD_METHOD_TO(ApiV1Controller::Health, "/api/v1/health", drogon::Get, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::ConvertDefaults, "/api/v1/convert/defaults", drogon::Get,
                  drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::VectorizeDefaults, "/api/v1/vectorize/defaults", drogon::Get,
                  drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::Databases, "/api/v1/databases", drogon::Get, drogon::Options);

    ADD_METHOD_TO(ApiV1Controller::SessionDatabases, "/api/v1/session/databases", drogon::Get,
                  drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::UploadSessionDatabase, "/api/v1/session/databases/upload",
                  drogon::Post, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::DeleteSessionDatabase, "/api/v1/session/databases/{1}",
                  drogon::Delete, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::DownloadSessionDatabase,
                  "/api/v1/session/databases/{1}/download", drogon::Get, drogon::Options);

    ADD_METHOD_TO(ApiV1Controller::SubmitConvertRaster, "/api/v1/convert/raster", drogon::Post,
                  drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::SubmitConvertRasterMatchOnly,
                  "/api/v1/convert/raster/match-only", drogon::Post, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::SubmitConvertVector, "/api/v1/convert/vector", drogon::Post,
                  drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::SubmitConvertVectorMatchOnly,
                  "/api/v1/convert/vector/match-only", drogon::Post, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::AnalyzeVectorWidth, "/api/v1/convert/vector/analyze-width",
                  drogon::Post, drogon::Options);

    ADD_METHOD_TO(ApiV1Controller::RecipeEditorSummary, "/api/v1/tasks/{1}/recipe-editor/summary",
                  drogon::Get, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::RecipeEditorAlternatives,
                  "/api/v1/tasks/{1}/recipe-editor/alternatives", drogon::Post, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::RecipeEditorReplace, "/api/v1/tasks/{1}/recipe-editor/replace",
                  drogon::Post, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::RecipeEditorGenerate, "/api/v1/tasks/{1}/recipe-editor/generate",
                  drogon::Post, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::RecipeEditorPredict, "/api/v1/tasks/{1}/recipe-editor/predict",
                  drogon::Post, drogon::Options);

    ADD_METHOD_TO(ApiV1Controller::ModelPackInfo, "/api/v1/model-pack/info", drogon::Get,
                  drogon::Options);

    ADD_METHOD_TO(ApiV1Controller::MattingMethods, "/api/v1/matting/methods", drogon::Get,
                  drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::SubmitMatting, "/api/v1/matting/tasks", drogon::Post,
                  drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::PostprocessMatting, "/api/v1/matting/tasks/{1}/postprocess",
                  drogon::Post, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::SubmitVectorize, "/api/v1/vectorize/tasks", drogon::Post,
                  drogon::Options);

    ADD_METHOD_TO(ApiV1Controller::Tasks, "/api/v1/tasks", drogon::Get, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::TaskById, "/api/v1/tasks/{1}", drogon::Get, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::DeleteTask, "/api/v1/tasks/{1}", drogon::Delete,
                  drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::DownloadTaskArtifactRoute, "/api/v1/tasks/{1}/artifacts/{2}",
                  drogon::Get, drogon::Options);

    ADD_METHOD_TO(ApiV1Controller::GenerateBoard, "/api/v1/calibration/boards", drogon::Post,
                  drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::Generate8ColorBoard, "/api/v1/calibration/boards/8color",
                  drogon::Post, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::DownloadBoardModel, "/api/v1/calibration/boards/{1}/model",
                  drogon::Get, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::DownloadBoardMeta, "/api/v1/calibration/boards/{1}/meta",
                  drogon::Get, drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::LocateBoard, "/api/v1/calibration/locate", drogon::Post,
                  drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::BuildColorDb, "/api/v1/calibration/colordb", drogon::Post,
                  drogon::Options);

    ADD_METHOD_TO(ApiV1Controller::ListAnnouncements, "/api/v1/announcements", drogon::Get,
                  drogon::Options);
    ADD_METHOD_TO(ApiV1Controller::UpsertAnnouncement, "/api/v1/announcements", drogon::Post);
    ADD_METHOD_TO(ApiV1Controller::DeleteAnnouncement, "/api/v1/announcements/{1}", drogon::Delete);
    METHOD_LIST_END

    using Callback = std::function<void(const drogon::HttpResponsePtr&)>;

    void Health(const drogon::HttpRequestPtr& req, Callback&& cb);
    void ConvertDefaults(const drogon::HttpRequestPtr& req, Callback&& cb);
    void VectorizeDefaults(const drogon::HttpRequestPtr& req, Callback&& cb);
    void Databases(const drogon::HttpRequestPtr& req, Callback&& cb);

    void SessionDatabases(const drogon::HttpRequestPtr& req, Callback&& cb);
    void UploadSessionDatabase(const drogon::HttpRequestPtr& req, Callback&& cb);
    void DeleteSessionDatabase(const drogon::HttpRequestPtr& req, Callback&& cb,
                               const std::string& name);
    void DownloadSessionDatabase(const drogon::HttpRequestPtr& req, Callback&& cb,
                                 const std::string& name);

    void SubmitConvertRaster(const drogon::HttpRequestPtr& req, Callback&& cb);
    void SubmitConvertRasterMatchOnly(const drogon::HttpRequestPtr& req, Callback&& cb);
    void SubmitConvertVector(const drogon::HttpRequestPtr& req, Callback&& cb);
    void SubmitConvertVectorMatchOnly(const drogon::HttpRequestPtr& req, Callback&& cb);
    void AnalyzeVectorWidth(const drogon::HttpRequestPtr& req, Callback&& cb);

    void RecipeEditorSummary(const drogon::HttpRequestPtr& req, Callback&& cb,
                             const std::string& task_id);
    void RecipeEditorAlternatives(const drogon::HttpRequestPtr& req, Callback&& cb,
                                  const std::string& task_id);
    void RecipeEditorReplace(const drogon::HttpRequestPtr& req, Callback&& cb,
                             const std::string& task_id);
    void RecipeEditorGenerate(const drogon::HttpRequestPtr& req, Callback&& cb,
                              const std::string& task_id);
    void RecipeEditorPredict(const drogon::HttpRequestPtr& req, Callback&& cb,
                             const std::string& task_id);

    void ModelPackInfo(const drogon::HttpRequestPtr& req, Callback&& cb);
    void MattingMethods(const drogon::HttpRequestPtr& req, Callback&& cb);
    void SubmitMatting(const drogon::HttpRequestPtr& req, Callback&& cb);
    void PostprocessMatting(const drogon::HttpRequestPtr& req, Callback&& cb,
                            const std::string& task_id);
    void SubmitVectorize(const drogon::HttpRequestPtr& req, Callback&& cb);

    void Tasks(const drogon::HttpRequestPtr& req, Callback&& cb);
    void TaskById(const drogon::HttpRequestPtr& req, Callback&& cb, const std::string& id);
    void DeleteTask(const drogon::HttpRequestPtr& req, Callback&& cb, const std::string& id);
    void DownloadTaskArtifactRoute(const drogon::HttpRequestPtr& req, Callback&& cb,
                                   const std::string& id, const std::string& artifact);

    void GenerateBoard(const drogon::HttpRequestPtr& req, Callback&& cb);
    void Generate8ColorBoard(const drogon::HttpRequestPtr& req, Callback&& cb);
    void DownloadBoardModel(const drogon::HttpRequestPtr& req, Callback&& cb,
                            const std::string& id);
    void DownloadBoardMeta(const drogon::HttpRequestPtr& req, Callback&& cb, const std::string& id);
    void LocateBoard(const drogon::HttpRequestPtr& req, Callback&& cb);
    void BuildColorDb(const drogon::HttpRequestPtr& req, Callback&& cb);

    void ListAnnouncements(const drogon::HttpRequestPtr& req, Callback&& cb);
    void UpsertAnnouncement(const drogon::HttpRequestPtr& req, Callback&& cb);
    void DeleteAnnouncement(const drogon::HttpRequestPtr& req, Callback&& cb,
                            const std::string& id);

private:
    ServerFacade& Facade() const;

    void ApplySessionCookie(const drogon::HttpResponsePtr& resp, const std::string& token) const;
    std::optional<std::string> SessionToken(const drogon::HttpRequestPtr& req) const;

    void ReplyJson(Callback&& cb, const ServiceResult& result,
                   const std::optional<std::string>& set_session_token = std::nullopt);
    void ReplyBinary(Callback&& cb, const TaskArtifact& artifact,
                     const std::optional<std::string>& set_session_token = std::nullopt);

    static drogon::HttpStatusCode ToHttpStatus(int status_code);
};

} // namespace chromaprint3d::backend
