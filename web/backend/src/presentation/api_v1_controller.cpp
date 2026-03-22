#include "presentation/api_v1_controller.h"

#include "presentation/backend_runtime.h"

#include <drogon/Cookie.h>
#include <drogon/MultiPart.h>
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

namespace chromaprint3d::backend {
namespace {

constexpr const char* kSessionHeader = "X-ChromaPrint3D-Session";

std::vector<uint8_t> ToBytes(const drogon::HttpFile& file) {
    auto content = file.fileContent();
    return std::vector<uint8_t>(content.begin(), content.end());
}

const drogon::HttpFile* FindUploadFile(const drogon::MultiPartParser& parser,
                                       const std::string& item_name) {
    for (const auto& file : parser.getFiles()) {
        if (file.getItemName() == item_name) return &file;
    }
    return nullptr;
}

} // namespace

ServerFacade& ApiV1Controller::Facade() const { return GetBackendFacade(); }

void ApiV1Controller::Health(const drogon::HttpRequestPtr& /*req*/, Callback&& cb) {
    ReplyJson(std::move(cb), Facade().Health());
}

void ApiV1Controller::ConvertDefaults(const drogon::HttpRequestPtr& /*req*/, Callback&& cb) {
    ReplyJson(std::move(cb), Facade().ConvertDefaults());
}

void ApiV1Controller::VectorizeDefaults(const drogon::HttpRequestPtr& /*req*/, Callback&& cb) {
    ReplyJson(std::move(cb), Facade().VectorizeDefaults());
}

void ApiV1Controller::Databases(const drogon::HttpRequestPtr& req, Callback&& cb) {
    ReplyJson(std::move(cb), Facade().ListDatabases(SessionToken(req)));
}

void ApiV1Controller::SessionDatabases(const drogon::HttpRequestPtr& req, Callback&& cb) {
    auto token = SessionToken(req);
    if (!token) {
        ReplyJson(std::move(cb),
                  ServiceResult::Success(200, {{"databases", nlohmann::json::array()}}));
        return;
    }
    ReplyJson(std::move(cb), Facade().ListSessionDatabases(*token));
}

void ApiV1Controller::UploadSessionDatabase(const drogon::HttpRequestPtr& req, Callback&& cb) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_multipart", "Invalid multipart form"));
        return;
    }
    auto file = FindUploadFile(parser, "file");
    if (!file) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_request", "Missing required field: file"));
        return;
    }

    bool created = false;
    auto token   = Facade().EnsureSession(SessionToken(req), &created);
    std::optional<std::string> name;
    auto raw_name = parser.getOptionalParameter<std::string>("name");
    if (raw_name.has_value() && !raw_name->empty()) name = *raw_name;

    auto result = Facade().UploadSessionDatabase(token, std::string(file->fileContent()), name);
    ReplyJson(std::move(cb), result, created ? std::optional<std::string>(token) : std::nullopt);
}

void ApiV1Controller::DeleteSessionDatabase(const drogon::HttpRequestPtr& req, Callback&& cb,
                                            const std::string& name) {
    auto token = SessionToken(req);
    if (!token) {
        ReplyJson(std::move(cb), ServiceResult::Error(401, "unauthorized", "No session"));
        return;
    }
    ReplyJson(std::move(cb), Facade().DeleteSessionDatabase(*token, name));
}

void ApiV1Controller::DownloadSessionDatabase(const drogon::HttpRequestPtr& req, Callback&& cb,
                                              const std::string& name) {
    auto token = SessionToken(req);
    if (!token) {
        ReplyJson(std::move(cb), ServiceResult::Error(401, "unauthorized", "No session"));
        return;
    }

    std::string content;
    std::string filename;
    auto ret = Facade().DownloadSessionDatabase(*token, name, content, filename);
    if (!ret.ok) {
        ReplyJson(std::move(cb), ret);
        return;
    }
    TaskArtifact artifact;
    artifact.bytes.assign(content.begin(), content.end());
    artifact.content_type = "application/json";
    artifact.filename     = filename;
    ReplyBinary(std::move(cb), artifact);
}

void ApiV1Controller::SubmitConvertRaster(const drogon::HttpRequestPtr& req, Callback&& cb) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_multipart", "Invalid multipart form"));
        return;
    }
    auto image = FindUploadFile(parser, "image");
    if (!image) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_request", "Missing required field: image"));
        return;
    }
    auto params  = parser.getOptionalParameter<std::string>("params");
    bool created = false;
    auto token   = Facade().EnsureSession(SessionToken(req), &created);
    auto result =
        Facade().SubmitConvertRaster(token, ToBytes(*image), image->getFileName(), params);
    ReplyJson(std::move(cb), result, created ? std::optional<std::string>(token) : std::nullopt);
}

void ApiV1Controller::SubmitConvertRasterMatchOnly(const drogon::HttpRequestPtr& req,
                                                   Callback&& cb) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_multipart", "Invalid multipart form"));
        return;
    }
    auto image = FindUploadFile(parser, "image");
    if (!image) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_request", "Missing required field: image"));
        return;
    }
    auto params  = parser.getOptionalParameter<std::string>("params");
    bool created = false;
    auto token   = Facade().EnsureSession(SessionToken(req), &created);
    auto result =
        Facade().SubmitConvertRasterMatchOnly(token, ToBytes(*image), image->getFileName(), params);
    ReplyJson(std::move(cb), result, created ? std::optional<std::string>(token) : std::nullopt);
}

void ApiV1Controller::RecipeEditorSummary(const drogon::HttpRequestPtr& req, Callback&& cb,
                                          const std::string& task_id) {
    auto token = SessionToken(req);
    if (!token) {
        ReplyJson(std::move(cb), ServiceResult::Error(401, "unauthorized", "Session required"));
        return;
    }
    ReplyJson(std::move(cb), Facade().RecipeEditorSummary(*token, task_id));
}

void ApiV1Controller::RecipeEditorAlternatives(const drogon::HttpRequestPtr& req, Callback&& cb,
                                               const std::string& task_id) {
    auto token = SessionToken(req);
    if (!token) {
        ReplyJson(std::move(cb), ServiceResult::Error(401, "unauthorized", "Session required"));
        return;
    }
    std::string body(req->body());
    if (body.empty()) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_request", "Request body is empty"));
        return;
    }
    ReplyJson(std::move(cb), Facade().RecipeEditorAlternatives(*token, task_id, body));
}

void ApiV1Controller::RecipeEditorReplace(const drogon::HttpRequestPtr& req, Callback&& cb,
                                          const std::string& task_id) {
    auto token = SessionToken(req);
    if (!token) {
        ReplyJson(std::move(cb), ServiceResult::Error(401, "unauthorized", "Session required"));
        return;
    }
    std::string body(req->body());
    if (body.empty()) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_request", "Request body is empty"));
        return;
    }
    ReplyJson(std::move(cb), Facade().RecipeEditorReplace(*token, task_id, body));
}

void ApiV1Controller::RecipeEditorGenerate(const drogon::HttpRequestPtr& req, Callback&& cb,
                                           const std::string& task_id) {
    auto token = SessionToken(req);
    if (!token) {
        ReplyJson(std::move(cb), ServiceResult::Error(401, "unauthorized", "Session required"));
        return;
    }
    ReplyJson(std::move(cb), Facade().RecipeEditorGenerate(*token, task_id));
}

void ApiV1Controller::SubmitConvertVector(const drogon::HttpRequestPtr& req, Callback&& cb) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_multipart", "Invalid multipart form"));
        return;
    }
    auto svg = FindUploadFile(parser, "svg");
    if (!svg) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_request", "Missing required field: svg"));
        return;
    }
    auto params  = parser.getOptionalParameter<std::string>("params");
    bool created = false;
    auto token   = Facade().EnsureSession(SessionToken(req), &created);
    auto result  = Facade().SubmitConvertVector(token, ToBytes(*svg), svg->getFileName(), params);
    ReplyJson(std::move(cb), result, created ? std::optional<std::string>(token) : std::nullopt);
}

void ApiV1Controller::SubmitConvertVectorMatchOnly(const drogon::HttpRequestPtr& req,
                                                   Callback&& cb) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_multipart", "Invalid multipart form"));
        return;
    }
    auto svg = FindUploadFile(parser, "svg");
    if (!svg) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_request", "Missing required field: svg"));
        return;
    }
    auto params  = parser.getOptionalParameter<std::string>("params");
    bool created = false;
    auto token   = Facade().EnsureSession(SessionToken(req), &created);
    auto result =
        Facade().SubmitConvertVectorMatchOnly(token, ToBytes(*svg), svg->getFileName(), params);
    ReplyJson(std::move(cb), result, created ? std::optional<std::string>(token) : std::nullopt);
}

void ApiV1Controller::AnalyzeVectorWidth(const drogon::HttpRequestPtr& req, Callback&& cb) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_multipart", "Invalid multipart form"));
        return;
    }
    auto svg = FindUploadFile(parser, "svg");
    if (!svg) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_request", "Missing required field: svg"));
        return;
    }
    auto params = parser.getOptionalParameter<std::string>("params");
    ReplyJson(std::move(cb), Facade().AnalyzeVectorWidth(ToBytes(*svg), params));
}

void ApiV1Controller::MattingMethods(const drogon::HttpRequestPtr& /*req*/, Callback&& cb) {
    ReplyJson(std::move(cb), Facade().MattingMethods());
}

void ApiV1Controller::SubmitMatting(const drogon::HttpRequestPtr& req, Callback&& cb) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_multipart", "Invalid multipart form"));
        return;
    }
    auto image = FindUploadFile(parser, "image");
    if (!image) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_request", "Missing required field: image"));
        return;
    }
    std::string method = parser.getOptionalParameter<std::string>("method").value_or("opencv");
    bool created       = false;
    auto token         = Facade().EnsureSession(SessionToken(req), &created);
    auto result = Facade().SubmitMatting(token, ToBytes(*image), image->getFileName(), method);
    ReplyJson(std::move(cb), result, created ? std::optional<std::string>(token) : std::nullopt);
}

void ApiV1Controller::PostprocessMatting(const drogon::HttpRequestPtr& req, Callback&& cb,
                                         const std::string& task_id) {
    auto token = SessionToken(req);
    if (!token) {
        ReplyJson(std::move(cb), ServiceResult::Error(401, "unauthorized", "Session required"));
        return;
    }
    std::string body(req->body());
    if (body.empty()) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_request", "Request body is empty"));
        return;
    }
    ReplyJson(std::move(cb), Facade().PostprocessMatting(*token, task_id, body));
}

void ApiV1Controller::SubmitVectorize(const drogon::HttpRequestPtr& req, Callback&& cb) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) {
        spdlog::warn("API vectorize submit rejected: invalid multipart payload");
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_multipart", "Invalid multipart form"));
        return;
    }
    auto image = FindUploadFile(parser, "image");
    if (!image) {
        spdlog::warn("API vectorize submit rejected: missing image field");
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_request", "Missing required field: image"));
        return;
    }
    auto params            = parser.getOptionalParameter<std::string>("params");
    bool created           = false;
    auto token             = Facade().EnsureSession(SessionToken(req), &created);
    const auto image_bytes = image->fileContent().size();
    const auto params_bytes =
        params.has_value() && !params->empty() ? static_cast<std::size_t>(params->size()) : 0U;
    spdlog::info("API vectorize submit: image='{}', bytes={}, params_bytes={}, session_created={}",
                 image->getFileName(), image_bytes, params_bytes, created);
    auto result = Facade().SubmitVectorize(token, ToBytes(*image), image->getFileName(), params);
    std::string task_id = "-";
    if (result.ok && result.data.is_object() && result.data.contains("task_id") &&
        result.data["task_id"].is_string()) {
        task_id = result.data["task_id"].get<std::string>();
    }
    if (result.ok) {
        spdlog::info("API vectorize submit accepted: status={}, task_id={}", result.status_code,
                     task_id);
    } else {
        spdlog::warn("API vectorize submit failed: status={}, code={}, message={}",
                     result.status_code, result.code, result.message);
    }
    ReplyJson(std::move(cb), result, created ? std::optional<std::string>(token) : std::nullopt);
}

void ApiV1Controller::Tasks(const drogon::HttpRequestPtr& req, Callback&& cb) {
    ReplyJson(std::move(cb), Facade().ListTasks(SessionToken(req).value_or("")));
}

void ApiV1Controller::TaskById(const drogon::HttpRequestPtr& req, Callback&& cb,
                               const std::string& id) {
    auto token = SessionToken(req);
    if (!token) {
        ReplyJson(std::move(cb), ServiceResult::Error(401, "unauthorized", "No session"));
        return;
    }
    ReplyJson(std::move(cb), Facade().GetTask(*token, id));
}

void ApiV1Controller::DeleteTask(const drogon::HttpRequestPtr& req, Callback&& cb,
                                 const std::string& id) {
    auto token = SessionToken(req);
    if (!token) {
        ReplyJson(std::move(cb), ServiceResult::Error(401, "unauthorized", "No session"));
        return;
    }
    ReplyJson(std::move(cb), Facade().DeleteTask(*token, id));
}

void ApiV1Controller::DownloadTaskArtifactRoute(const drogon::HttpRequestPtr& req, Callback&& cb,
                                                const std::string& id,
                                                const std::string& artifact) {
    auto token = SessionToken(req);
    if (!token) {
        ReplyJson(std::move(cb), ServiceResult::Error(401, "unauthorized", "No session"));
        return;
    }
    TaskArtifact out;
    auto r = Facade().DownloadTaskArtifact(*token, id, artifact, out);
    if (!r.ok) {
        ReplyJson(std::move(cb), r);
        return;
    }
    ReplyBinary(std::move(cb), out);
}

void ApiV1Controller::GenerateBoard(const drogon::HttpRequestPtr& req, Callback&& cb) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req->body());
    } catch (...) {
        ReplyJson(std::move(cb), ServiceResult::Error(400, "invalid_json", "Expected JSON body"));
        return;
    }
    if (!body.is_object()) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_json", "Expected JSON object body"));
        return;
    }
    ReplyJson(std::move(cb), Facade().GenerateBoard(body));
}

void ApiV1Controller::Generate8ColorBoard(const drogon::HttpRequestPtr& req, Callback&& cb) {
    nlohmann::json body;
    try {
        body = nlohmann::json::parse(req->body());
    } catch (...) {
        ReplyJson(std::move(cb), ServiceResult::Error(400, "invalid_json", "Expected JSON body"));
        return;
    }
    if (!body.is_object()) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_json", "Expected JSON object body"));
        return;
    }
    ReplyJson(std::move(cb), Facade().Generate8ColorBoard(body));
}

void ApiV1Controller::DownloadBoardModel(const drogon::HttpRequestPtr& /*req*/, Callback&& cb,
                                         const std::string& id) {
    TaskArtifact out;
    auto r = Facade().DownloadBoardModel(id, out);
    if (!r.ok) {
        ReplyJson(std::move(cb), r);
        return;
    }
    ReplyBinary(std::move(cb), out);
}

void ApiV1Controller::DownloadBoardMeta(const drogon::HttpRequestPtr& /*req*/, Callback&& cb,
                                        const std::string& id) {
    TaskArtifact out;
    auto r = Facade().DownloadBoardMeta(id, out);
    if (!r.ok) {
        ReplyJson(std::move(cb), r);
        return;
    }
    ReplyBinary(std::move(cb), out);
}

void ApiV1Controller::LocateBoard(const drogon::HttpRequestPtr& req, Callback&& cb) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_multipart", "Invalid multipart form"));
        return;
    }
    auto image = FindUploadFile(parser, "image");
    auto meta  = FindUploadFile(parser, "meta");
    if (!image || !meta) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_request", "Missing image/meta upload"));
        return;
    }
    auto result = Facade().LocateBoard(ToBytes(*image), std::string(meta->fileContent()));
    ReplyJson(std::move(cb), result);
}

void ApiV1Controller::BuildColorDb(const drogon::HttpRequestPtr& req, Callback&& cb) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_multipart", "Invalid multipart form"));
        return;
    }
    auto image = FindUploadFile(parser, "image");
    auto meta  = FindUploadFile(parser, "meta");
    if (!image || !meta) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_request", "Missing image/meta upload"));
        return;
    }
    auto name = parser.getOptionalParameter<std::string>("name").value_or("");
    if (name.empty()) {
        ReplyJson(std::move(cb),
                  ServiceResult::Error(400, "invalid_request", "Missing required field: name"));
        return;
    }
    auto corners_json = parser.getOptionalParameter<std::string>("corners").value_or("");
    bool created      = false;
    auto token        = Facade().EnsureSession(SessionToken(req), &created);
    auto result = Facade().BuildColorDb(token, ToBytes(*image), std::string(meta->fileContent()),
                                        name, corners_json);
    ReplyJson(std::move(cb), result, created ? std::optional<std::string>(token) : std::nullopt);
}

void ApiV1Controller::ApplySessionCookie(const drogon::HttpResponsePtr& resp,
                                         const std::string& token) const {
    drogon::Cookie cookie("session", token);
    cookie.setHttpOnly(true);
    cookie.setPath("/");
    cookie.setMaxAge(static_cast<int>(Facade().Config().session_ttl_seconds));
    if (!Facade().Config().cors_origin.empty()) {
        cookie.setSameSite(drogon::Cookie::SameSite::kNone);
        cookie.setSecure(true);
    } else {
        cookie.setSameSite(drogon::Cookie::SameSite::kStrict);
    }
    resp->addCookie(std::move(cookie));
}

std::optional<std::string> ApiV1Controller::SessionToken(const drogon::HttpRequestPtr& req) const {
    auto token = req->getCookie("session");
    if (token.empty()) token = req->getHeader(kSessionHeader);
    if (token.empty()) return std::nullopt;
    return token;
}

void ApiV1Controller::ReplyJson(Callback&& cb, const ServiceResult& result,
                                const std::optional<std::string>& set_session_token) {
    nlohmann::json body;
    if (result.ok) {
        body["ok"]   = true;
        body["data"] = result.data;
    } else {
        body["ok"]    = false;
        body["error"] = {
            {"code", result.code.empty() ? "error" : result.code},
            {"message", result.message},
        };
    }
    auto resp = drogon::HttpResponse::newHttpResponse();
    resp->setBody(body.dump());
    resp->setContentTypeString("application/json; charset=utf-8");
    resp->setStatusCode(ToHttpStatus(result.status_code));
    if (set_session_token) {
        ApplySessionCookie(resp, *set_session_token);
        resp->addHeader(kSessionHeader, *set_session_token);
    }
    cb(resp);
}

void ApiV1Controller::ReplyBinary(Callback&& cb, const TaskArtifact& artifact,
                                  const std::optional<std::string>& set_session_token) {
    drogon::HttpResponsePtr resp;

    if (artifact.is_file_based()) {
        spdlog::debug("ReplyBinary: serving file artifact: path={}, filename={}, content_type={}",
                      artifact.file_path.string(), artifact.filename, artifact.content_type);
        resp =
            drogon::HttpResponse::newFileResponse(artifact.file_path.string(), artifact.filename);
        resp->setContentTypeString(artifact.content_type);
    } else {
        spdlog::debug("ReplyBinary: serving buffer artifact: size={}, filename={}, content_type={}",
                      artifact.bytes.size(), artifact.filename, artifact.content_type);
        resp = drogon::HttpResponse::newHttpResponse();
        resp->setStatusCode(drogon::k200OK);
        resp->setBody(std::string(reinterpret_cast<const char*>(artifact.bytes.data()),
                                  artifact.bytes.size()));
        resp->setContentTypeString(artifact.content_type);
        if (!artifact.filename.empty()) {
            resp->addHeader("Content-Disposition",
                            "attachment; filename=\"" + artifact.filename + "\"");
        }
    }

    if (set_session_token) {
        ApplySessionCookie(resp, *set_session_token);
        resp->addHeader(kSessionHeader, *set_session_token);
    }
    cb(resp);
}

drogon::HttpStatusCode ApiV1Controller::ToHttpStatus(int status_code) {
    if (status_code >= 100 && status_code <= 599) {
        return static_cast<drogon::HttpStatusCode>(status_code);
    }
    return drogon::k500InternalServerError;
}

} // namespace chromaprint3d::backend
