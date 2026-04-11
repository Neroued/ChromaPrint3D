#pragma once

#include "application/service_result.h"
#include "application/request_parsing.h"
#include "application/colordb_resolution.h"
#include "config/server_config.h"
#include "infrastructure/data_repository.h"
#include "infrastructure/session_store.h"
#include "infrastructure/task_runtime.h"
#include "chromaprint3d/pipeline.h"

#include <nlohmann/json.hpp>

#include <optional>
#include <string>
#include <vector>

namespace chromaprint3d::backend {

class ConvertService {
public:
    ConvertService(const ServerConfig& cfg, DataRepository& data, SessionStore& sessions,
                   TaskRuntime& tasks);

    ServiceResult SubmitConvertRaster(const std::string& owner, const std::vector<uint8_t>& image,
                                      const std::string& image_name,
                                      const std::optional<std::string>& params_json);
    ServiceResult SubmitConvertVector(const std::string& owner, const std::vector<uint8_t>& svg,
                                      const std::string& svg_name,
                                      const std::optional<std::string>& params_json);
    ServiceResult AnalyzeVectorWidth(const std::vector<uint8_t>& svg,
                                     const std::optional<std::string>& params_json);
    ServiceResult SubmitConvertRasterMatchOnly(const std::string& owner,
                                               const std::vector<uint8_t>& image,
                                               const std::string& image_name,
                                               const std::optional<std::string>& params_json);
    ServiceResult SubmitConvertVectorMatchOnly(const std::string& owner,
                                               const std::vector<uint8_t>& svg,
                                               const std::string& svg_name,
                                               const std::optional<std::string>& params_json);

private:
    ServiceResult BuildRasterRequest(const nlohmann::json& params,
                                     const std::vector<uint8_t>& image,
                                     const std::string& image_name,
                                     const std::optional<SessionSnapshot>& session,
                                     ChromaPrint3D::ConvertRasterRequest& out) const;
    ServiceResult BuildVectorRequest(const nlohmann::json& params, const std::vector<uint8_t>& svg,
                                     const std::string& svg_name,
                                     const std::optional<SessionSnapshot>& session,
                                     ChromaPrint3D::ConvertVectorRequest& out) const;

    const ServerConfig& cfg_;
    DataRepository& data_;
    SessionStore& sessions_;
    TaskRuntime& tasks_;
};

} // namespace chromaprint3d::backend
