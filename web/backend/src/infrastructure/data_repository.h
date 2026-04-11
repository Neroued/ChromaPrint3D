#pragma once

#include "config/server_config.h"
#include "server/color_db_cache.h"
#include "server/recipe_store.h"

#include "chromaprint3d/forward_color_model.h"
#include "chromaprint3d/matting.h"
#include "chromaprint3d/model_package.h"
#if defined(CHROMAPRINT3D_HAS_INFER) && CHROMAPRINT3D_HAS_INFER
#    include "chromaprint3d/infer/engine.h"
#endif

#include <memory>
#include <optional>
#include <string>

namespace chromaprint3d::backend {

class DataRepository {
public:
    explicit DataRepository(const ServerConfig& cfg);

    const ColorDBCache& ColorDbCache() const { return db_cache_; }

    const ChromaPrint3D::ModelPackageRegistry& ModelPackRegistry() const { return model_packs_; }

    const EightColorRecipeStore& RecipeStore() const { return recipe_store_; }

    const ChromaPrint3D::MattingRegistry& MattingRegistry() const { return matting_registry_; }

    const ChromaPrint3D::ForwardModelRegistry& ForwardModels() const { return forward_models_; }

private:
    ColorDBCache db_cache_;
    ChromaPrint3D::ModelPackageRegistry model_packs_;
    ChromaPrint3D::ForwardModelRegistry forward_models_;
    EightColorRecipeStore recipe_store_;
    ChromaPrint3D::MattingRegistry matting_registry_;
#if defined(CHROMAPRINT3D_HAS_INFER) && CHROMAPRINT3D_HAS_INFER
    // Keep infer engine alive for the full DataRepository lifetime.
    std::unique_ptr<ChromaPrint3D::infer::InferenceEngine> infer_engine_;
#endif
};

} // namespace chromaprint3d::backend
