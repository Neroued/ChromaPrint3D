#pragma once

#include "application/service_result.h"
#include "infrastructure/task_runtime.h"

#include <string>

namespace chromaprint3d::backend {

class TaskService {
public:
    explicit TaskService(TaskRuntime& tasks);

    ServiceResult ListTasks(const std::string& owner) const;
    ServiceResult GetTask(const std::string& owner, const std::string& id);
    ServiceResult DeleteTask(const std::string& owner, const std::string& id);
    ServiceResult DownloadTaskArtifact(const std::string& owner, const std::string& id,
                                       const std::string& artifact, TaskArtifact& out);

private:
    TaskRuntime& tasks_;
};

} // namespace chromaprint3d::backend
