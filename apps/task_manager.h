#pragma once

#include "chromaprint3d/pipeline.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace ChromaPrint3D {

struct TaskInfo {
    enum class Status : uint8_t { Pending, Running, Completed, Failed };

    std::string id;
    std::string image_name;
    Status status      = Status::Pending;
    ConvertStage stage = ConvertStage::LoadingResources;
    float progress     = 0.0f;
    std::string error_message;
    ConvertResult result;

    std::chrono::steady_clock::time_point created_at;
    std::chrono::steady_clock::time_point completed_at;
};

inline const char* TaskStatusToString(TaskInfo::Status s) {
    switch (s) {
    case TaskInfo::Status::Pending:
        return "pending";
    case TaskInfo::Status::Running:
        return "running";
    case TaskInfo::Status::Completed:
        return "completed";
    case TaskInfo::Status::Failed:
        return "failed";
    }
    return "unknown";
}

inline const char* ConvertStageToString(ConvertStage s) {
    switch (s) {
    case ConvertStage::LoadingResources:
        return "loading_resources";
    case ConvertStage::ProcessingImage:
        return "processing_image";
    case ConvertStage::Matching:
        return "matching";
    case ConvertStage::BuildingModel:
        return "building_model";
    case ConvertStage::Exporting:
        return "exporting";
    }
    return "unknown";
}

class TaskManager {
public:
    TaskManager(int max_concurrent, int task_ttl_seconds)
        : max_concurrent_(max_concurrent), running_count_(0), task_ttl_seconds_(task_ttl_seconds),
          shutdown_(false) {
        cleanup_thread_ = std::thread([this]() { CleanupLoop(); });
    }

    ~TaskManager() {
        shutdown_.store(true);
        cleanup_cv_.notify_all();
        if (cleanup_thread_.joinable()) { cleanup_thread_.join(); }

        // Wait for all worker threads to finish
        std::unique_lock<std::mutex> lock(mutex_);
        for (auto& t : worker_threads_) {
            lock.unlock();
            if (t.joinable()) { t.join(); }
            lock.lock();
        }
        worker_threads_.clear();
    }

    TaskManager(const TaskManager&)            = delete;
    TaskManager& operator=(const TaskManager&) = delete;

    std::string Submit(ConvertRequest request) {
        std::lock_guard<std::mutex> lock(mutex_);

        // Clean up finished worker threads
        CleanWorkerThreads();

        std::string id = GenerateId();
        TaskInfo info;
        info.id         = id;
        info.image_name = request.image_name;
        info.status     = TaskInfo::Status::Pending;
        info.created_at = std::chrono::steady_clock::now();
        tasks_.emplace(id, std::move(info));

        worker_threads_.emplace_back(
            [this, id, req = std::move(request)]() mutable { RunTask(id, std::move(req)); });

        return id;
    }

    std::optional<TaskInfo> GetTask(const std::string& id) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tasks_.find(id);
        if (it == tasks_.end()) { return std::nullopt; }
        // Return a copy without the heavy result buffers for status queries
        return it->second;
    }

    // Get a shared pointer to the task's result data (avoids copying large buffers)
    const std::vector<uint8_t>* GetTaskResultBuffer(const std::string& id,
                                                    const std::string& field) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tasks_.find(id);
        if (it == tasks_.end()) { return nullptr; }
        if (it->second.status != TaskInfo::Status::Completed) { return nullptr; }
        if (field == "model_3mf") { return &it->second.result.model_3mf; }
        if (field == "preview_png") { return &it->second.result.preview_png; }
        if (field == "source_mask_png") { return &it->second.result.source_mask_png; }
        return nullptr;
    }

    bool DeleteTask(const std::string& id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = tasks_.find(id);
        if (it == tasks_.end()) { return false; }
        // Only delete if not currently running
        if (it->second.status == TaskInfo::Status::Running) { return false; }
        tasks_.erase(it);
        return true;
    }

    int ActiveTaskCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return running_count_;
    }

    int TotalTaskCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return static_cast<int>(tasks_.size());
    }

    // Return lightweight summaries (without result buffers)
    std::vector<TaskInfo> ListTasks() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<TaskInfo> result;
        result.reserve(tasks_.size());
        for (const auto& [id, info] : tasks_) {
            TaskInfo summary;
            summary.id                  = info.id;
            summary.status              = info.status;
            summary.stage               = info.stage;
            summary.progress            = info.progress;
            summary.error_message       = info.error_message;
            summary.result.stats        = info.result.stats;
            summary.result.image_width  = info.result.image_width;
            summary.result.image_height = info.result.image_height;
            summary.created_at          = info.created_at;
            summary.completed_at        = info.completed_at;
            result.push_back(std::move(summary));
        }
        return result;
    }

private:
    void RunTask(const std::string& id, ConvertRequest request) {
        // Wait for a slot if at max concurrency
        {
            std::unique_lock<std::mutex> lock(mutex_);
            concurrency_cv_.wait(
                lock, [this]() { return running_count_ < max_concurrent_ || shutdown_.load(); });
            if (shutdown_.load()) { return; }

            auto it = tasks_.find(id);
            if (it == tasks_.end()) { return; }
            it->second.status = TaskInfo::Status::Running;
            ++running_count_;
        }

        ProgressCallback progress_cb = [this, &id](ConvertStage stage, float progress) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = tasks_.find(id);
            if (it != tasks_.end()) {
                it->second.stage    = stage;
                it->second.progress = progress;
            }
        };

        try {
            ConvertResult result = Convert(request, progress_cb);
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = tasks_.find(id);
            if (it != tasks_.end()) {
                it->second.status       = TaskInfo::Status::Completed;
                it->second.result       = std::move(result);
                it->second.completed_at = std::chrono::steady_clock::now();
                it->second.progress     = 1.0f;
            }
        } catch (const std::exception& e) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = tasks_.find(id);
            if (it != tasks_.end()) {
                it->second.status        = TaskInfo::Status::Failed;
                it->second.error_message = e.what();
                it->second.completed_at  = std::chrono::steady_clock::now();
            }
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = tasks_.find(id);
            if (it != tasks_.end()) {
                it->second.status        = TaskInfo::Status::Failed;
                it->second.error_message = "Unknown error";
                it->second.completed_at  = std::chrono::steady_clock::now();
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            --running_count_;
        }
        concurrency_cv_.notify_one();
    }

    void CleanupLoop() {
        while (!shutdown_.load()) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cleanup_cv_.wait_for(lock, std::chrono::seconds(60),
                                     [this]() { return shutdown_.load(); });
                if (shutdown_.load()) { return; }

                auto now = std::chrono::steady_clock::now();
                auto ttl = std::chrono::seconds(task_ttl_seconds_);
                std::vector<std::string> expired;
                for (const auto& [id, info] : tasks_) {
                    if (info.status == TaskInfo::Status::Completed ||
                        info.status == TaskInfo::Status::Failed) {
                        if (now - info.completed_at > ttl) { expired.push_back(id); }
                    }
                }
                for (const std::string& id : expired) { tasks_.erase(id); }
            }
        }
    }

    void CleanWorkerThreads() {
        // Must be called with mutex_ held
        auto it = worker_threads_.begin();
        while (it != worker_threads_.end()) {
            // Try to join finished threads
            // We can't easily check if a thread is done without joining,
            // so we only clean up joinable threads that have their task completed/failed
            it = std::next(it);
        }
        // Simple approach: move finished threads out
        std::vector<std::thread> active;
        for (auto& t : worker_threads_) { active.push_back(std::move(t)); }
        worker_threads_ = std::move(active);
    }

    std::string GenerateId() {
        static thread_local std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<uint64_t> dist;
        uint64_t a = dist(gen);
        uint64_t b = dist(gen);

        static const char hex[] = "0123456789abcdef";
        std::string id;
        id.reserve(32);
        for (int i = 15; i >= 0; --i) { id.push_back(hex[(a >> (i * 4)) & 0xf]); }
        for (int i = 15; i >= 0; --i) { id.push_back(hex[(b >> (i * 4)) & 0xf]); }
        return id;
    }

    mutable std::mutex mutex_;
    std::condition_variable concurrency_cv_;
    std::condition_variable cleanup_cv_;
    std::unordered_map<std::string, TaskInfo> tasks_;
    std::vector<std::thread> worker_threads_;

    int max_concurrent_;
    int running_count_;
    int task_ttl_seconds_;
    std::atomic<bool> shutdown_;
    std::thread cleanup_thread_;
};

} // namespace ChromaPrint3D
