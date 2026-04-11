#pragma once

#include "application/service_result.h"
#include "infrastructure/data_repository.h"
#include "infrastructure/session_store.h"

#include <optional>
#include <string>

namespace chromaprint3d::backend {

class ColorDbService {
public:
    explicit ColorDbService(DataRepository& data, SessionStore& sessions);

    ServiceResult ListDatabases(const std::optional<std::string>& session_token);
    ServiceResult ListSessionDatabases(const std::string& token);
    ServiceResult UploadSessionDatabase(const std::string& token, const std::string& json_content,
                                        const std::optional<std::string>& override_name);
    ServiceResult DeleteSessionDatabase(const std::string& token, const std::string& name);
    ServiceResult DownloadSessionDatabase(const std::string& token, const std::string& name,
                                          std::string& json_out, std::string& filename_out);

private:
    DataRepository& data_;
    SessionStore& sessions_;
};

} // namespace chromaprint3d::backend
