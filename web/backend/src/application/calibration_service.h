#pragma once

#include "application/service_result.h"
#include "config/server_config.h"
#include "infrastructure/board_runtime_cache.h"
#include "infrastructure/data_repository.h"
#include "infrastructure/session_store.h"
#include "infrastructure/task_runtime.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace chromaprint3d::backend {

class CalibrationService {
public:
    CalibrationService(const ServerConfig& cfg, DataRepository& data, SessionStore& sessions,
                       BoardRuntimeCache& boards);

    ServiceResult GenerateBoard(const nlohmann::json& body);
    ServiceResult Generate8ColorBoard(const nlohmann::json& body);
    ServiceResult DownloadBoardModel(const std::string& board_id, TaskArtifact& out);
    ServiceResult DownloadBoardMeta(const std::string& board_id, TaskArtifact& out);
    ServiceResult LocateBoard(const std::vector<uint8_t>& image, const std::string& meta_json);
    ServiceResult BuildColorDb(const std::string& owner, const std::vector<uint8_t>& image,
                               const std::string& meta_json, const std::string& db_name,
                               const std::string& corners_json = "");

private:
    const ServerConfig& cfg_;
    DataRepository& data_;
    SessionStore& sessions_;
    BoardRuntimeCache& boards_;
};

} // namespace chromaprint3d::backend
