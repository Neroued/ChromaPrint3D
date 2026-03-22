#pragma once

#include "infrastructure/spillable_artifact.h"

#include "chromaprint3d/matting.h"
#include "chromaprint3d/matting_postprocess.h"
#include "chromaprint3d/pipeline.h"
#include "chromaprint3d/raster_region_map.h"
#include "chromaprint3d/recipe_alternatives.h"

#include <neroued/vectorizer/vectorizer.h>
#include <nlohmann/json.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

namespace chromaprint3d::backend {

enum class TaskKind { Convert, Matting, Vectorize };
enum class RuntimeTaskStatus : std::uint8_t { Pending, Running, Completed, Failed };

const char* TaskKindToString(TaskKind kind);
const char* TaskStatusToString(RuntimeTaskStatus status);

struct ConvertTaskPayload {
    std::string input_name;
    ChromaPrint3D::ConvertStage stage = ChromaPrint3D::ConvertStage::LoadingResources;
    float progress                    = 0.0f;
    ChromaPrint3D::ConvertResult result;
    bool has_3mf_on_disk = false;

    bool match_only = false;
    std::string generate_error;
    std::optional<ChromaPrint3D::MatchRasterResult> raster_match_state;
    std::optional<ChromaPrint3D::RasterRegionMap> raster_region_map;
    std::vector<uint8_t> region_map_binary;

    std::optional<ChromaPrint3D::MatchVectorResult> vector_match_state;

    ChromaPrint3D::RecipeSearchCache recipe_search_cache;
};

struct MattingTaskPayload {
    std::string method;
    std::string image_name;

    ChromaPrint3D::MattingTimingInfo timing;
    double decode_ms   = 0;
    double encode_ms   = 0;
    double pipeline_ms = 0;

    std::vector<uint8_t> mask_png;
    std::vector<uint8_t> foreground_png;
    int width  = 0;
    int height = 0;

    std::vector<uint8_t> alpha_png;
    std::vector<uint8_t> original_png;
    bool has_alpha = false;

    std::vector<uint8_t> processed_mask_png;
    std::vector<uint8_t> processed_fg_png;
    std::vector<uint8_t> outline_png;
};

struct VectorizeTaskPayload {
    std::string image_name;
    neroued::vectorizer::VectorizerConfig config;

    double decode_ms    = 0;
    double vectorize_ms = 0;
    double pipeline_ms  = 0;

    std::string svg_content;
    std::size_t svg_bytes   = 0;
    bool has_svg_on_disk    = false;
    int width               = 0;
    int height              = 0;
    int num_shapes          = 0;
    int resolved_num_colors = 0;
};

using TaskPayload = std::variant<ConvertTaskPayload, MattingTaskPayload, VectorizeTaskPayload>;

struct TaskSnapshot {
    std::string id;
    std::string owner;
    TaskKind kind            = TaskKind::Convert;
    RuntimeTaskStatus status = RuntimeTaskStatus::Pending;
    std::string error;

    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point completed_at;
    TaskPayload payload;
};

struct TaskArtifact {
    std::vector<uint8_t> bytes;
    std::filesystem::path file_path;
    std::string content_type;
    std::string filename;

    bool is_file_based() const { return !file_path.empty(); }
};

struct SubmitResult {
    bool ok         = false;
    int status_code = 500;
    std::string task_id;
    std::string message;
};

class TaskRuntime {
public:
    TaskRuntime(std::int64_t worker_count, std::int64_t max_queue, std::int64_t ttl_seconds,
                std::int64_t max_tasks_per_owner, std::int64_t max_total_result_bytes,
                const std::string& data_dir);
    ~TaskRuntime();

    TaskRuntime(const TaskRuntime&)            = delete;
    TaskRuntime& operator=(const TaskRuntime&) = delete;

    SubmitResult SubmitConvertRaster(const std::string& owner,
                                     ChromaPrint3D::ConvertRasterRequest req,
                                     const std::string& input_name);
    SubmitResult SubmitConvertVector(const std::string& owner,
                                     ChromaPrint3D::ConvertVectorRequest req,
                                     const std::string& input_name);
    SubmitResult SubmitMatting(const std::string& owner, std::vector<uint8_t> image_buffer,
                               const std::string& method, const std::string& image_name,
                               const ChromaPrint3D::MattingRegistry& registry);
    SubmitResult SubmitVectorize(const std::string& owner, std::vector<uint8_t> image_buffer,
                                 neroued::vectorizer::VectorizerConfig config,
                                 const std::string& image_name);

    SubmitResult SubmitConvertRasterMatchOnly(const std::string& owner,
                                              ChromaPrint3D::ConvertRasterRequest req,
                                              const std::string& input_name);
    SubmitResult SubmitConvertVectorMatchOnly(const std::string& owner,
                                              ChromaPrint3D::ConvertVectorRequest req,
                                              const std::string& input_name);

    /// Lightweight admission check before expensive decode/preprocess.
    SubmitResult CanAccept(const std::string& owner) const;

    /// Synchronously apply post-processing to a completed matting task.
    /// Returns false on error (sets status_code and message).
    bool PostprocessMatting(const std::string& owner, const std::string& id,
                            const ChromaPrint3D::MattingPostprocessParams& params, int& status_code,
                            std::string& message);

    std::optional<nlohmann::json> GetRecipeEditorSummary(const std::string& owner,
                                                         const std::string& id) const;

    std::optional<nlohmann::json>
    QueryRecipeAlternatives(const std::string& owner, const std::string& id,
                            const ChromaPrint3D::Lab& target_lab, int max_candidates, int offset,
                            const ChromaPrint3D::ModelPackage* model_pack);

    bool ReplaceRecipe(const std::string& owner, const std::string& id,
                       const std::vector<int>& target_region_ids,
                       const std::vector<uint8_t>& new_recipe,
                       const ChromaPrint3D::Lab& new_mapped_color, bool new_from_model,
                       int& status_code, std::string& message, nlohmann::json& out_summary);

    SubmitResult SubmitGenerateModel(const std::string& owner, const std::string& id,
                                     const ChromaPrint3D::ModelPackage* model_pack);

    std::vector<TaskSnapshot> ListTasks(const std::string& owner) const;
    std::optional<TaskSnapshot> FindTask(const std::string& owner, const std::string& id) const;
    bool DeleteTask(const std::string& owner, const std::string& id, int& status_code,
                    std::string& message);
    std::optional<TaskArtifact> LoadArtifact(const std::string& owner, const std::string& id,
                                             const std::string& artifact, int& status_code,
                                             std::string& message) const;

    int ActiveTaskCount() const;

    int TotalTaskCount() const { return total_submitted_.load(std::memory_order_relaxed); }

private:
    using WorkerFn = std::function<void(const std::string& task_id)>;

    struct TaskRecord {
        TaskSnapshot snapshot;
        std::size_t artifact_bytes = 0;
        SpillableArtifact spilled_3mf;
        SpillableArtifact spilled_svg;
    };

    struct QueueEntry {
        std::string task_id;
        WorkerFn worker;
    };

    SubmitResult SubmitInternal(const std::string& owner, TaskKind kind, TaskPayload payload,
                                WorkerFn worker);
    void RunTask(const std::string& id, WorkerFn worker);
    void StartWorkers(std::int64_t worker_count);
    void WorkerLoop();

    void MarkRunning(const std::string& id);
    void MarkFailed(const std::string& id, const std::string& message);
    void MarkCompleted(const std::string& id);

    template <typename Fn>
    void UpdateTaskPayload(const std::string& id, Fn&& fn) {
        std::lock_guard<std::mutex> lock(task_mtx_);
        auto it = tasks_.find(id);
        if (it == tasks_.end()) return;
        fn(it->second.snapshot.payload);
    }

    template <typename Fn>
    void UpdateTaskRecord(const std::string& id, Fn&& fn) {
        std::lock_guard<std::mutex> lock(task_mtx_);
        auto it = tasks_.find(id);
        if (it == tasks_.end()) return;
        fn(it->second);
    }

    std::size_t ComputeArtifactBytes(const TaskSnapshot& snapshot) const;
    std::optional<nlohmann::json> GetRecipeEditorSummaryLocked(const TaskRecord& rec) const;
    void CleanupLoop();
    void CleanupExpiredLocked(const std::chrono::steady_clock::time_point& now);
    void EnforceResultBudgetLocked();
    std::size_t ActiveTasksByOwnerLocked(const std::string& owner) const;
    static std::string NewTaskId();

    std::int64_t max_queue_;
    std::int64_t ttl_seconds_;
    std::int64_t max_tasks_per_owner_;
    std::int64_t max_total_result_bytes_;
    std::filesystem::path temp_dir_;

    mutable std::mutex task_mtx_;
    std::unordered_map<std::string, TaskRecord> tasks_;
    std::size_t total_artifact_bytes_ = 0;

    mutable std::mutex queue_mtx_;
    std::condition_variable queue_cv_;
    std::deque<QueueEntry> queue_;
    std::vector<std::thread> workers_;

    std::atomic<bool> shutdown_{false};
    std::thread cleanup_thread_;
    std::atomic<int> running_count_{0};
    std::atomic<int> total_submitted_{0};
};

} // namespace chromaprint3d::backend
