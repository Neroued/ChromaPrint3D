#pragma once

#include "application/service_result.h"
#include "infrastructure/data_repository.h"
#include "infrastructure/session_store.h"

#include "chromaprint3d/color_db.h"

#include <nlohmann/json.hpp>

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace chromaprint3d::backend {

std::string NormalizeChannelKey(const ChromaPrint3D::Channel& ch);
std::vector<std::string>
BuildProfileChannelKeys(const std::vector<const ChromaPrint3D::ColorDB*>& dbs);

ServiceResult
ResolveSelectedColorDbs(const nlohmann::json& params, const std::optional<SessionSnapshot>& session,
                        const DataRepository& data,
                        std::vector<const ChromaPrint3D::ColorDB*>& out_dbs,
                        std::vector<std::shared_ptr<const ChromaPrint3D::ColorDB>>& session_owned,
                        std::string& common_vendor, std::string& common_material);

void ResolveTaskVendorMaterial(const std::vector<std::string>& db_names, const DataRepository& data,
                               std::string& common_vendor, std::string& common_material);

} // namespace chromaprint3d::backend
