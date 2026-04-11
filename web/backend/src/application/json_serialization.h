#pragma once

#include "infrastructure/task_runtime.h"

#include "chromaprint3d/color_db.h"
#include "chromaprint3d/pipeline.h"

#include <nlohmann/json.hpp>

#include <string>
#include <vector>

namespace chromaprint3d::backend {

const char* ClusterMethodToString(ChromaPrint3D::ClusterMethod method);
const char* LayerOrderToString(ChromaPrint3D::LayerOrder order);
const char* ConvertStageToString(ChromaPrint3D::ConvertStage stage);

nlohmann::json LayerPreviewsToJson(const ChromaPrint3D::LayerPreviewResult& layer_previews);

nlohmann::json ColorDbInfoToJson(const ChromaPrint3D::ColorDB& db,
                                 const std::string& material_type = "",
                                 const std::string& vendor        = "");
nlohmann::json ColorDbBuildResultToJson(const ChromaPrint3D::ColorDB& db);

nlohmann::json TaskToJson(const TaskSnapshot& task);

} // namespace chromaprint3d::backend
