#include "infrastructure/data_repository.h"

#if defined(CHROMAPRINT3D_HAS_INFER) && CHROMAPRINT3D_HAS_INFER
#    include <chromaprint3d/infer/infer.h>
#endif

#include <filesystem>

#include <spdlog/spdlog.h>

namespace chromaprint3d::backend {
namespace {

ColorDBCache LoadColorDBs(const std::filesystem::path& data_root) {
    auto dbs_dir = data_root / "dbs";
    if (!std::filesystem::is_directory(dbs_dir)) dbs_dir = data_root;

    spdlog::info("Loading ColorDBs from: {}", dbs_dir.string());
    ColorDBCache cache;
    cache.LoadFromDirectory(dbs_dir.string());
    spdlog::info("Loaded {} ColorDB(s)", cache.databases.size());
    return cache;
}

EightColorRecipeStore TryLoadRecipes(const std::filesystem::path& data_root) {
    auto recipe_path = data_root / "recipes" / "8color_boards.json";
    if (!std::filesystem::is_regular_file(recipe_path)) {
        spdlog::info("No 8-color recipe file found at {}", recipe_path.string());
        return {};
    }
    try {
        return EightColorRecipeStore::LoadFromFile(recipe_path.string());
    } catch (const std::exception& e) {
        spdlog::warn("Failed to load 8-color recipes: {}", e.what());
        return {};
    }
}

#if defined(CHROMAPRINT3D_HAS_INFER) && CHROMAPRINT3D_HAS_INFER
void LoadMattingModels(ChromaPrint3D::MattingRegistry& registry,
                       ChromaPrint3D::infer::InferenceEngine& engine,
                       const std::filesystem::path& model_dir) {
    auto discovered = ChromaPrint3D::DiscoverMattingModels(model_dir.string());
    if (discovered.empty()) return;

    ChromaPrint3D::infer::SessionOptions sess_opts;
    if (engine.SupportsDevice(ChromaPrint3D::infer::DeviceType::kCUDA)) {
        sess_opts.device = ChromaPrint3D::infer::Device::CUDA();
        spdlog::info("CUDA available, matting models will use GPU");
    } else {
        sess_opts.device = ChromaPrint3D::infer::Device::CPU();
    }

    for (auto& [model_path, config] : discovered) {
        std::string stem = std::filesystem::path(model_path).stem().string();
        try {
            auto session = engine.LoadModel(model_path, sess_opts);
            auto provider =
                ChromaPrint3D::CreateDLMattingProvider(stem, std::move(session), config);
            registry.Register(stem, std::move(provider));
        } catch (const std::exception& e) {
            spdlog::error("Failed to load matting model '{}': {}", stem, e.what());
        }
    }
}
#endif

void LogMattingProviders(const ChromaPrint3D::MattingRegistry& registry) {
    auto keys = registry.Available();
    std::string joined;
    for (const auto& k : keys) {
        if (!joined.empty()) joined += ", ";
        joined += k;
    }
    spdlog::info("Matting providers: [{}]", joined);
}

} // namespace

DataRepository::DataRepository(const ServerConfig& cfg) {
    auto data_root = std::filesystem::path(cfg.data_dir);
    db_cache_      = LoadColorDBs(data_root);

    if (cfg.model_pack_path.empty()) {
        model_packs_.LoadFromDirectory((data_root / "model_packs").string());
    } else if (std::filesystem::is_directory(cfg.model_pack_path)) {
        model_packs_.LoadFromDirectory(cfg.model_pack_path);
    } else {
        model_packs_.LoadSingle(cfg.model_pack_path);
    }

    forward_models_.LoadFromDirectory((data_root / "models" / "forward").string());

    recipe_store_ = TryLoadRecipes(data_root);

    matting_registry_.Register("opencv",
                               std::make_shared<ChromaPrint3D::ThresholdMattingProvider>());

    auto model_dir = data_root / "models" / "matting";
#if defined(CHROMAPRINT3D_HAS_INFER) && CHROMAPRINT3D_HAS_INFER
    infer_engine_ = std::make_unique<ChromaPrint3D::infer::InferenceEngine>(
        ChromaPrint3D::infer::InferenceEngine::Create());
    LoadMattingModels(matting_registry_, *infer_engine_, model_dir);
#else
    auto discovered = ChromaPrint3D::DiscoverMattingModels(model_dir.string());
    if (!discovered.empty()) {
        spdlog::warn("Found {} matting model(s) but infer module is unavailable",
                     discovered.size());
    }
#endif
    LogMattingProviders(matting_registry_);
}

} // namespace chromaprint3d::backend
