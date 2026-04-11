#pragma once

#include "application/service_result.h"

#include "infrastructure/data_repository.h"
#include "infrastructure/task_runtime.h"

#include <string>

namespace chromaprint3d::backend {

class RecipeEditorService {
public:
    RecipeEditorService(DataRepository& data, TaskRuntime& tasks);

    ServiceResult RecipeEditorSummary(const std::string& owner, const std::string& task_id);
    ServiceResult RecipeEditorAlternatives(const std::string& owner, const std::string& task_id,
                                           const std::string& body_json);
    ServiceResult RecipeEditorReplace(const std::string& owner, const std::string& task_id,
                                      const std::string& body_json);
    ServiceResult RecipeEditorGenerate(const std::string& owner, const std::string& task_id);
    ServiceResult RecipeEditorPredict(const std::string& owner, const std::string& task_id,
                                      const std::string& body_json);

private:
    DataRepository& data_;
    TaskRuntime& tasks_;
};

} // namespace chromaprint3d::backend
