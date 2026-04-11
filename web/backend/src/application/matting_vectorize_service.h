#pragma once

#include "application/service_result.h"
#include "application/request_parsing.h"
#include "config/server_config.h"
#include "infrastructure/data_repository.h"
#include "infrastructure/task_runtime.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace neroued::vectorizer {
struct VectorizerConfig;
}

namespace chromaprint3d::backend {

class MattingVectorizeService {
public:
    MattingVectorizeService(const ServerConfig& cfg, DataRepository& data, TaskRuntime& tasks);

    ServiceResult SubmitMatting(const std::string& owner, const std::vector<uint8_t>& image,
                                const std::string& image_name, const std::string& method);
    ServiceResult PostprocessMatting(const std::string& owner, const std::string& task_id,
                                     const std::string& body_json);
    ServiceResult SubmitVectorize(const std::string& owner, const std::vector<uint8_t>& image,
                                  const std::string& image_name,
                                  const std::optional<std::string>& params_json);

private:
    ServiceResult BuildVectorizeConfig(const nlohmann::json& params,
                                       neroued::vectorizer::VectorizerConfig& out) const;

    const ServerConfig& cfg_;
    DataRepository& data_;
    TaskRuntime& tasks_;
};

} // namespace chromaprint3d::backend
