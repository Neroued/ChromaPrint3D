#include "application/task_service.h"

#include "application/json_serialization.h"

namespace chromaprint3d::backend {

using json = nlohmann::json;

TaskService::TaskService(TaskRuntime& tasks) : tasks_(tasks) {}

ServiceResult TaskService::ListTasks(const std::string& owner) const {
    if (owner.empty()) return ServiceResult::Success(200, {{"tasks", json::array()}});
    json tasks = json::array();
    for (const auto& t : tasks_.ListTasks(owner)) tasks.push_back(TaskToJson(t));
    return ServiceResult::Success(200, {{"tasks", std::move(tasks)}});
}

ServiceResult TaskService::GetTask(const std::string& owner, const std::string& id) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    auto task = tasks_.FindTask(owner, id);
    if (!task) return ServiceResult::Error(404, "not_found", "Task not found");
    return ServiceResult::Success(200, TaskToJson(*task));
}

ServiceResult TaskService::DeleteTask(const std::string& owner, const std::string& id) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    int status          = 500;
    std::string message = "Unknown error";
    bool ok             = tasks_.DeleteTask(owner, id, status, message);
    if (!ok) return ServiceResult::Error(status, "delete_failed", message);
    return ServiceResult::Success(200, {{"deleted", true}});
}

ServiceResult TaskService::DownloadTaskArtifact(const std::string& owner, const std::string& id,
                                                const std::string& artifact, TaskArtifact& out) {
    if (owner.empty()) return ServiceResult::Error(401, "unauthorized", "No session");
    int status          = 500;
    std::string message = "Unknown error";
    auto data           = tasks_.LoadArtifact(owner, id, artifact, status, message);
    if (!data) return ServiceResult::Error(status, "artifact_error", message);
    out = std::move(*data);
    return ServiceResult::Success(200, json::object());
}

} // namespace chromaprint3d::backend
