#include <gtest/gtest.h>

#include "application/task_service.h"
#include "chromaprint3d/color_db.h"
#include "chromaprint3d/pipeline.h"
#include "infrastructure/task_runtime.h"

#include <opencv2/imgcodecs.hpp>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace chromaprint3d::backend {
namespace {

using namespace std::chrono_literals;
using ChromaPrint3D::ColorDB;
using ChromaPrint3D::ConvertRasterRequest;

std::filesystem::path MakeUniqueTempDir() {
    static std::atomic<unsigned long long> counter{0};
    const auto suffix =
        std::to_string(counter.fetch_add(1, std::memory_order_relaxed) +
                       static_cast<unsigned long long>(
                           std::chrono::steady_clock::now().time_since_epoch().count()));
    const auto dir =
        std::filesystem::temp_directory_path() / ("chromaprint3d-task-runtime-" + suffix);
    std::filesystem::create_directories(dir);
    return dir;
}

std::shared_ptr<const ColorDB> LoadTestColorDb() {
    static const auto db = std::make_shared<const ColorDB>(
        ColorDB::LoadFromJson((std::filesystem::path(CHROMAPRINT3D_TEST_DATA_DIR) / "dbs" / "PLA" /
                               "BambuLab" / "RYBW_008_5L.json")
                                  .string()));
    return db;
}

std::vector<uint8_t> EncodeTestPng() {
    cv::Mat image(8, 8, CV_8UC3, cv::Scalar(32, 64, 192));
    std::vector<uint8_t> bytes;
    if (!cv::imencode(".png", image, bytes)) {
        throw std::runtime_error("failed to encode test PNG");
    }
    return bytes;
}

const ConvertTaskPayload& RequireConvertPayload(const TaskSnapshot& snapshot) {
    const auto* payload = std::get_if<ConvertTaskPayload>(&snapshot.payload);
    if (!payload) throw std::runtime_error("task payload is not convert");
    return *payload;
}

template <typename Predicate>
TaskSnapshot WaitForSnapshot(TaskRuntime& runtime, const std::string& owner, const std::string& id,
                             Predicate&& predicate, std::chrono::milliseconds timeout = 5s) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        auto snapshot = runtime.FindTask(owner, id);
        if (snapshot && predicate(*snapshot)) { return *snapshot; }
        std::this_thread::sleep_for(10ms);
    }
    throw std::runtime_error("timeout waiting for task snapshot");
}

class SharedTaskRuntimeHarness {
public:
    SharedTaskRuntimeHarness()
        : data_dir_(MakeUniqueTempDir()),
          runtime_(std::make_unique<TaskRuntime>(1, 16, 3600, 16, 256 * 1024 * 1024,
                                                 data_dir_.string())),
          db_(LoadTestColorDb()) {}

    TaskRuntime& runtime() { return *runtime_; }

    std::filesystem::path temp_results_dir() const { return data_dir_ / "tmp" / "results"; }

    SubmitResult SubmitRasterMatchOnly(const std::string& owner, const std::string& input_name) {
        ConvertRasterRequest request;
        request.image_buffer         = EncodeTestPng();
        request.image_name           = input_name + ".png";
        request.preloaded_dbs        = {db_};
        request.color_layers         = 5;
        request.generate_preview     = true;
        request.generate_source_mask = true;
        request.model_enable         = false;
        request.max_width            = 8;
        request.max_height           = 8;
        return runtime_->SubmitConvertRasterMatchOnly(owner, std::move(request), input_name);
    }

private:
    std::filesystem::path data_dir_;
    std::unique_ptr<TaskRuntime> runtime_;
    std::shared_ptr<const ColorDB> db_;
};

SharedTaskRuntimeHarness& GetSharedHarness() {
    static auto* harness = new SharedTaskRuntimeHarness();
    return *harness;
}

TEST(TaskRuntimeTest, MatchOnlyGenerationFailurePreservesTaskStateAndAllowsRetry) {
    auto& harness           = GetSharedHarness();
    auto& runtime           = harness.runtime();
    const std::string owner = "runtime-retry-owner";

    const auto submit = harness.SubmitRasterMatchOnly(owner, "retryable-match-only");
    ASSERT_TRUE(submit.ok) << submit.message;

    const auto matched =
        WaitForSnapshot(runtime, owner, submit.task_id, [](const TaskSnapshot& snapshot) {
            return snapshot.status == RuntimeTaskStatus::Completed ||
                   snapshot.status == RuntimeTaskStatus::Failed;
        });
    ASSERT_EQ(matched.status, RuntimeTaskStatus::Completed);

    const auto initial_completed_at = matched.completed_at;
    const auto& initial_payload     = RequireConvertPayload(matched);
    EXPECT_TRUE(initial_payload.match_only);
    EXPECT_EQ(initial_payload.generation.status, GenerationStatus::Idle);
    EXPECT_FALSE(initial_payload.result.preview_png.empty());
    EXPECT_FALSE(initial_payload.result.source_mask_png.empty());

    std::error_code ec;
    std::filesystem::remove_all(harness.temp_results_dir(), ec);
    ASSERT_FALSE(ec) << ec.message();

    const auto first_generate = runtime.SubmitGenerateModel(owner, submit.task_id);
    ASSERT_TRUE(first_generate.ok) << first_generate.message;

    const auto failed =
        WaitForSnapshot(runtime, owner, submit.task_id, [](const TaskSnapshot& snapshot) {
            const auto& payload = RequireConvertPayload(snapshot);
            return payload.generation.status == GenerationStatus::Failed;
        });

    EXPECT_EQ(failed.status, RuntimeTaskStatus::Completed);
    EXPECT_EQ(failed.completed_at, initial_completed_at);

    const auto& failed_payload = RequireConvertPayload(failed);
    EXPECT_EQ(failed_payload.generation.status, GenerationStatus::Failed);
    EXPECT_FALSE(failed_payload.generation.error.empty());
    EXPECT_FALSE(failed_payload.has_3mf_on_disk);
    EXPECT_TRUE(runtime.GetRecipeEditorSummary(owner, submit.task_id).has_value());

    TaskService task_service(runtime);
    const auto failed_task_json = task_service.GetTask(owner, submit.task_id);
    ASSERT_TRUE(failed_task_json.ok) << failed_task_json.message;
    EXPECT_EQ(failed_task_json.data.at("status"), "completed");
    EXPECT_EQ(failed_task_json.data.at("generation").at("status"), "failed");
    EXPECT_FALSE(failed_task_json.data.at("result").at("has_3mf").get<bool>());

    std::filesystem::create_directories(harness.temp_results_dir(), ec);
    ASSERT_FALSE(ec) << ec.message();

    const auto retry_generate = runtime.SubmitGenerateModel(owner, submit.task_id);
    ASSERT_TRUE(retry_generate.ok) << retry_generate.message;

    const auto succeeded = WaitForSnapshot(
        runtime, owner, submit.task_id,
        [](const TaskSnapshot& snapshot) {
            const auto& payload = RequireConvertPayload(snapshot);
            return payload.generation.status == GenerationStatus::Succeeded;
        },
        15s);

    EXPECT_EQ(succeeded.status, RuntimeTaskStatus::Completed);
    EXPECT_EQ(succeeded.completed_at, initial_completed_at);

    const auto& succeeded_payload = RequireConvertPayload(succeeded);
    EXPECT_EQ(succeeded_payload.generation.status, GenerationStatus::Succeeded);
    EXPECT_TRUE(succeeded_payload.generation.error.empty());
    EXPECT_TRUE(succeeded_payload.has_3mf_on_disk || !succeeded_payload.result.model_3mf.empty());
    EXPECT_FALSE(succeeded_payload.result.layer_previews.layer_pngs.empty());

    const auto succeeded_task_json = task_service.GetTask(owner, submit.task_id);
    ASSERT_TRUE(succeeded_task_json.ok) << succeeded_task_json.message;
    EXPECT_EQ(succeeded_task_json.data.at("status"), "completed");
    EXPECT_EQ(succeeded_task_json.data.at("generation").at("status"), "succeeded");
    EXPECT_TRUE(succeeded_task_json.data.at("result").at("has_3mf").get<bool>());

    TaskArtifact downloaded_artifact;
    const auto download_result =
        task_service.DownloadTaskArtifact(owner, submit.task_id, "result", downloaded_artifact);
    ASSERT_TRUE(download_result.ok) << download_result.message;
    EXPECT_EQ(download_result.status_code, 200);
    EXPECT_TRUE(downloaded_artifact.is_file_based() || !downloaded_artifact.bytes.empty());
}

TEST(TaskRuntimeTest, FindTaskRefreshesLastAccessedWithoutMovingCompletedAt) {
    auto& harness           = GetSharedHarness();
    auto& runtime           = harness.runtime();
    const std::string owner = "runtime-access-owner";

    const auto submit = harness.SubmitRasterMatchOnly(owner, "access-tracking");
    ASSERT_TRUE(submit.ok) << submit.message;

    const auto matched =
        WaitForSnapshot(runtime, owner, submit.task_id, [](const TaskSnapshot& snapshot) {
            return snapshot.status == RuntimeTaskStatus::Completed ||
                   snapshot.status == RuntimeTaskStatus::Failed;
        });
    ASSERT_EQ(matched.status, RuntimeTaskStatus::Completed);

    const auto first = runtime.FindTask(owner, submit.task_id);
    ASSERT_TRUE(first.has_value());

    std::this_thread::sleep_for(15ms);

    const auto second = runtime.FindTask(owner, submit.task_id);
    ASSERT_TRUE(second.has_value());

    EXPECT_EQ(second->completed_at, first->completed_at);
    EXPECT_GT(second->last_accessed_at, first->last_accessed_at);
}

} // namespace
} // namespace chromaprint3d::backend
