#include "application/server_facade.h"

#include "application/json_serialization.h"

#include "chromaprint3d/memory_utils.h"
#include "chromaprint3d/pipeline.h"
#include "chromaprint3d/version.h"

namespace chromaprint3d::backend {

using json = nlohmann::json;
using namespace ChromaPrint3D;
using neroued::vectorizer::VectorizerConfig;

// ---------------------------------------------------------------------------
// Construction & session
// ---------------------------------------------------------------------------

ServerFacade::ServerFacade(ServerConfig cfg)
    : cfg_(std::move(cfg)), data_(cfg_),
      sessions_(cfg_.session_ttl_seconds, cfg_.max_session_colordbs),
      tasks_(cfg_.max_tasks, cfg_.max_task_queue, cfg_.task_ttl_seconds, cfg_.max_tasks_per_owner,
             cfg_.max_task_result_mb * 1024 * 1024, cfg_.data_dir),
      boards_(cfg_.board_cache_ttl, cfg_.data_dir, cfg_.board_geometry_cache_ttl,
              cfg_.board_geometry_cache_max),
      color_db_svc_(data_, sessions_), task_svc_(tasks_),
      convert_svc_(cfg_, data_, sessions_, tasks_), matting_svc_(cfg_, data_, tasks_),
      recipe_svc_(data_, tasks_), calib_svc_(cfg_, data_, sessions_, boards_),
      announcement_svc_(cfg_.data_dir) {}

std::string ServerFacade::EnsureSession(const std::optional<std::string>& existing_token,
                                        bool* created) {
    return sessions_.EnsureSession(existing_token, created);
}

std::optional<SessionSnapshot> ServerFacade::SessionSnapshotByToken(const std::string& token) {
    return sessions_.Snapshot(token);
}

// ---------------------------------------------------------------------------
// Cross-domain queries (kept in facade)
// ---------------------------------------------------------------------------

ServiceResult ServerFacade::Health() const {
    const auto heap   = ChromaPrint3D::GetHeapStats();
    const auto budget = tasks_.GetArtifactBudgetInfo();

    json memory = {
        {"rss_bytes", ChromaPrint3D::GetProcessRssBytes()},
        {"heap_allocated_bytes", heap.valid ? heap.allocated : 0},
        {"heap_resident_bytes", heap.valid ? heap.resident : 0},
        {"artifact_budget_bytes", budget.used_bytes},
        {"artifact_budget_limit_bytes", budget.limit_bytes},
        {"colordb_pool_bytes",
         data_.ColorDbCache().EstimateTotalBytes() + sessions_.EstimateAllSessionDbBytes()},
        {"memory_limit_bytes", ChromaPrint3D::GetMemoryLimitBytes()},
        {"allocator", ChromaPrint3D::AllocatorName()},
    };

    return ServiceResult::Success(
        200, {
                 {"status", "ok"},
                 {"version", CHROMAPRINT3D_VERSION_STRING},
                 {"active_tasks", tasks_.ActiveTaskCount()},
                 {"total_tasks", tasks_.TotalTaskCount()},
                 {"memory", std::move(memory)},
                 {"announcements_version", announcement_svc_.VersionHash()},
                 {"active_announcement_count", announcement_svc_.ActiveCount()},
             });
}

ServiceResult ServerFacade::ConvertDefaults() const {
    ConvertRasterRequest defaults;
    return ServiceResult::Success(
        200, {
                 {"scale", defaults.scale},
                 {"max_width", defaults.max_width},
                 {"max_height", defaults.max_height},
                 {"target_width_mm", defaults.target_width_mm},
                 {"target_height_mm", defaults.target_height_mm},
                 {"color_layers", defaults.color_layers},
                 {"color_space", "lab"},
                 {"k_candidates", defaults.k_candidates},
                 {"cluster_method", ClusterMethodToString(defaults.cluster_method)},
                 {"cluster_count", defaults.cluster_count},
                 {"slic_target_superpixels", defaults.slic_target_superpixels},
                 {"slic_compactness", defaults.slic_compactness},
                 {"slic_iterations", defaults.slic_iterations},
                 {"slic_min_region_ratio", defaults.slic_min_region_ratio},
                 {"dither", "none"},
                 {"dither_strength", defaults.dither_strength},
                 {"model_enable", defaults.model_enable},
                 {"model_only", defaults.model_only},
                 {"model_threshold", defaults.model_threshold},
                 {"model_margin", defaults.model_margin},
                 {"flip_y", defaults.flip_y},
                 {"pixel_mm", defaults.pixel_mm},
                 {"layer_height_mm", defaults.layer_height_mm},
                 {"base_layers", nullptr},
                 {"double_sided", defaults.double_sided},
                 {"transparent_layer_mm", defaults.transparent_layer_mm},
                 {"generate_preview", defaults.generate_preview},
                 {"generate_source_mask", defaults.generate_source_mask},
             });
}

ServiceResult ServerFacade::VectorizeDefaults() const {
    VectorizerConfig cfg;
    return ServiceResult::Success(200, {
                                           {"num_colors", cfg.num_colors},
                                           {"min_region_area", cfg.min_region_area},
                                           {"curve_fit_error", cfg.curve_fit_error},
                                           {"corner_angle_threshold", cfg.corner_angle_threshold},
                                           {"smoothing_spatial", cfg.smoothing_spatial},
                                           {"smoothing_color", cfg.smoothing_color},
                                           {"upscale_short_edge", cfg.upscale_short_edge},
                                           {"max_working_pixels", cfg.max_working_pixels},
                                           {"slic_region_size", cfg.slic_region_size},
                                           {"edge_sensitivity", cfg.edge_sensitivity},
                                           {"refine_passes", cfg.refine_passes},
                                           {"max_merge_color_dist", cfg.max_merge_color_dist},
                                           {"thin_line_max_radius", cfg.thin_line_max_radius},
                                           {"min_contour_area", cfg.min_contour_area},
                                           {"min_hole_area", cfg.min_hole_area},
                                           {"contour_simplify", cfg.contour_simplify},
                                           {"enable_coverage_fix", cfg.enable_coverage_fix},
                                           {"min_coverage_ratio", cfg.min_coverage_ratio},
                                           {"smoothness", cfg.smoothness},
                                           {"detail_level", cfg.detail_level},
                                           {"merge_segment_tolerance", cfg.merge_segment_tolerance},
                                           {"enable_antialias_detect", cfg.enable_antialias_detect},
                                           {"aa_tolerance", cfg.aa_tolerance},
                                           {"svg_enable_stroke", cfg.svg_enable_stroke},
                                           {"svg_stroke_width", cfg.svg_stroke_width},
                                       });
}

json ServerFacade::GetModelPackInfo() const {
    json packs = json::array();
    for (const auto& pack : data_.ModelPackRegistry().All()) {
        json modes = json::array();
        for (const auto& lp : pack->layer_packages) {
            modes.push_back({
                {"color_layers", lp.color_layers},
                {"layer_height_mm", lp.layer_height_mm},
                {"num_candidates", lp.NumCandidates()},
            });
        }
        packs.push_back({
            {"name", pack->name},
            {"schema_version", ModelPackage::kSchemaVersion},
            {"scope",
             {{"vendor", pack->scope.vendor}, {"material_type", pack->scope.material_type}}},
            {"channel_keys", pack->channel_keys},
            {"modes", std::move(modes)},
            {"defaults",
             {{"threshold", pack->default_threshold}, {"margin", pack->default_margin}}},
        });
    }
    return {{"packs", std::move(packs)}};
}

ServiceResult ServerFacade::MattingMethods() const {
    json arr = json::array();
    for (const auto& key : data_.MattingRegistry().Available()) {
        auto* provider = data_.MattingRegistry().Get(key);
        arr.push_back({
            {"key", key},
            {"name", provider ? provider->Name() : key},
            {"description", provider ? provider->Description() : ""},
        });
    }
    return ServiceResult::Success(200, {{"methods", std::move(arr)}});
}

// ---------------------------------------------------------------------------
// ColorDB delegation
// ---------------------------------------------------------------------------

ServiceResult ServerFacade::ListDatabases(const std::optional<std::string>& session_token) {
    return color_db_svc_.ListDatabases(session_token);
}

ServiceResult ServerFacade::ListSessionDatabases(const std::string& token) {
    return color_db_svc_.ListSessionDatabases(token);
}

ServiceResult ServerFacade::UploadSessionDatabase(const std::string& token,
                                                  const std::string& json_content,
                                                  const std::optional<std::string>& override_name) {
    return color_db_svc_.UploadSessionDatabase(token, json_content, override_name);
}

ServiceResult ServerFacade::DeleteSessionDatabase(const std::string& token,
                                                  const std::string& name) {
    return color_db_svc_.DeleteSessionDatabase(token, name);
}

ServiceResult ServerFacade::DownloadSessionDatabase(const std::string& token,
                                                    const std::string& name, std::string& json_out,
                                                    std::string& filename_out) {
    return color_db_svc_.DownloadSessionDatabase(token, name, json_out, filename_out);
}

// ---------------------------------------------------------------------------
// Convert delegation
// ---------------------------------------------------------------------------

ServiceResult ServerFacade::SubmitConvertRaster(const std::string& owner,
                                                const std::vector<uint8_t>& image,
                                                const std::string& image_name,
                                                const std::optional<std::string>& params_json) {
    return convert_svc_.SubmitConvertRaster(owner, image, image_name, params_json);
}

ServiceResult ServerFacade::SubmitConvertVector(const std::string& owner,
                                                const std::vector<uint8_t>& svg,
                                                const std::string& svg_name,
                                                const std::optional<std::string>& params_json) {
    return convert_svc_.SubmitConvertVector(owner, svg, svg_name, params_json);
}

ServiceResult ServerFacade::AnalyzeVectorWidth(const std::vector<uint8_t>& svg,
                                               const std::optional<std::string>& params_json) {
    return convert_svc_.AnalyzeVectorWidth(svg, params_json);
}

ServiceResult ServerFacade::SubmitConvertRasterMatchOnly(
    const std::string& owner, const std::vector<uint8_t>& image, const std::string& image_name,
    const std::optional<std::string>& params_json) {
    return convert_svc_.SubmitConvertRasterMatchOnly(owner, image, image_name, params_json);
}

ServiceResult ServerFacade::SubmitConvertVectorMatchOnly(
    const std::string& owner, const std::vector<uint8_t>& svg, const std::string& svg_name,
    const std::optional<std::string>& params_json) {
    return convert_svc_.SubmitConvertVectorMatchOnly(owner, svg, svg_name, params_json);
}

// ---------------------------------------------------------------------------
// Matting & vectorize delegation
// ---------------------------------------------------------------------------

ServiceResult ServerFacade::SubmitMatting(const std::string& owner,
                                          const std::vector<uint8_t>& image,
                                          const std::string& image_name,
                                          const std::string& method) {
    return matting_svc_.SubmitMatting(owner, image, image_name, method);
}

ServiceResult ServerFacade::PostprocessMatting(const std::string& owner, const std::string& task_id,
                                               const std::string& body_json) {
    return matting_svc_.PostprocessMatting(owner, task_id, body_json);
}

ServiceResult ServerFacade::SubmitVectorize(const std::string& owner,
                                            const std::vector<uint8_t>& image,
                                            const std::string& image_name,
                                            const std::optional<std::string>& params_json) {
    return matting_svc_.SubmitVectorize(owner, image, image_name, params_json);
}

// ---------------------------------------------------------------------------
// Recipe editor delegation
// ---------------------------------------------------------------------------

ServiceResult ServerFacade::RecipeEditorSummary(const std::string& owner,
                                                const std::string& task_id) {
    return recipe_svc_.RecipeEditorSummary(owner, task_id);
}

ServiceResult ServerFacade::RecipeEditorAlternatives(const std::string& owner,
                                                     const std::string& task_id,
                                                     const std::string& body_json) {
    return recipe_svc_.RecipeEditorAlternatives(owner, task_id, body_json);
}

ServiceResult ServerFacade::RecipeEditorReplace(const std::string& owner,
                                                const std::string& task_id,
                                                const std::string& body_json) {
    return recipe_svc_.RecipeEditorReplace(owner, task_id, body_json);
}

ServiceResult ServerFacade::RecipeEditorGenerate(const std::string& owner,
                                                 const std::string& task_id) {
    return recipe_svc_.RecipeEditorGenerate(owner, task_id);
}

ServiceResult ServerFacade::RecipeEditorPredict(const std::string& owner,
                                                const std::string& task_id,
                                                const std::string& body_json) {
    return recipe_svc_.RecipeEditorPredict(owner, task_id, body_json);
}

// ---------------------------------------------------------------------------
// Task delegation
// ---------------------------------------------------------------------------

ServiceResult ServerFacade::ListTasks(const std::string& owner) const {
    return task_svc_.ListTasks(owner);
}

ServiceResult ServerFacade::GetTask(const std::string& owner, const std::string& id) {
    return task_svc_.GetTask(owner, id);
}

ServiceResult ServerFacade::DeleteTask(const std::string& owner, const std::string& id) {
    return task_svc_.DeleteTask(owner, id);
}

ServiceResult ServerFacade::DownloadTaskArtifact(const std::string& owner, const std::string& id,
                                                 const std::string& artifact, TaskArtifact& out) {
    return task_svc_.DownloadTaskArtifact(owner, id, artifact, out);
}

// ---------------------------------------------------------------------------
// Calibration delegation
// ---------------------------------------------------------------------------

ServiceResult ServerFacade::GenerateBoard(const json& body) {
    return calib_svc_.GenerateBoard(body);
}

ServiceResult ServerFacade::Generate8ColorBoard(const json& body) {
    return calib_svc_.Generate8ColorBoard(body);
}

ServiceResult ServerFacade::DownloadBoardModel(const std::string& board_id, TaskArtifact& out) {
    return calib_svc_.DownloadBoardModel(board_id, out);
}

ServiceResult ServerFacade::DownloadBoardMeta(const std::string& board_id, TaskArtifact& out) {
    return calib_svc_.DownloadBoardMeta(board_id, out);
}

ServiceResult ServerFacade::LocateBoard(const std::vector<uint8_t>& image,
                                        const std::string& meta_json) {
    return calib_svc_.LocateBoard(image, meta_json);
}

ServiceResult ServerFacade::BuildColorDb(const std::string& owner,
                                         const std::vector<uint8_t>& image,
                                         const std::string& meta_json, const std::string& db_name,
                                         const std::string& corners_json) {
    return calib_svc_.BuildColorDb(owner, image, meta_json, db_name, corners_json);
}

// ---------------------------------------------------------------------------
// Announcements delegation
// ---------------------------------------------------------------------------

ServiceResult ServerFacade::ListAnnouncements() const {
    const auto items = announcement_svc_.ListActive();
    json arr         = json::array();
    for (const auto& a : items) { arr.push_back(AnnouncementService::ToWireJson(a)); }
    return ServiceResult::Success(200,
                                  {
                                      {"announcements", std::move(arr)},
                                      {"announcements_version", announcement_svc_.VersionHash()},
                                  });
}

ServiceResult ServerFacade::UpsertAnnouncement(const nlohmann::json& body) {
    return announcement_svc_.Upsert(body);
}

ServiceResult ServerFacade::DeleteAnnouncement(const std::string& id) {
    return announcement_svc_.Remove(id);
}

} // namespace chromaprint3d::backend
