#include "infrastructure/task_runtime.h"
#include "infrastructure/random_utils.h"

#include "chromaprint3d/color.h"
#include "chromaprint3d/encoding.h"
#include "chromaprint3d/image_io.h"
#include "chromaprint3d/matting_postprocess.h"
#include "chromaprint3d/vector_preview.h"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace {

std::string LabToHex(const ChromaPrint3D::Lab& lab) {
    auto rgb   = lab.ToRgb();
    uint8_t r8 = 0, g8 = 0, b8 = 0;
    rgb.ToRgb255(r8, g8, b8);
    std::ostringstream hex;
    hex << '#' << std::hex << std::setfill('0') << std::setw(2) << static_cast<int>(r8)
        << std::setw(2) << static_cast<int>(g8) << std::setw(2) << static_cast<int>(b8);
    return hex.str();
}

cv::Mat CompositeForeground(const cv::Mat& bgr, const cv::Mat& mask) {
    cv::Mat bgra;
    cv::cvtColor(bgr, bgra, cv::COLOR_BGR2BGRA);
    std::vector<cv::Mat> channels(4);
    cv::split(bgra, channels);
    channels[3] = mask;
    cv::merge(channels, bgra);
    return bgra;
}

constexpr const char* kLayerPreviewArtifactPrefix = "layer-preview-";

std::string OwnerTag(const std::string& owner) {
    if (owner.empty()) return "none";
    constexpr std::size_t kPrefix = 8;
    if (owner.size() <= kPrefix) return owner;
    return owner.substr(0, kPrefix);
}

double ElapsedMs(const std::chrono::steady_clock::time_point& begin,
                 const std::chrono::steady_clock::time_point& end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

} // namespace

namespace chromaprint3d::backend {

const char* TaskKindToString(TaskKind kind) {
    switch (kind) {
    case TaskKind::Convert:
        return "convert";
    case TaskKind::Matting:
        return "matting";
    case TaskKind::Vectorize:
        return "vectorize";
    }
    return "unknown";
}

const char* TaskStatusToString(RuntimeTaskStatus status) {
    switch (status) {
    case RuntimeTaskStatus::Pending:
        return "pending";
    case RuntimeTaskStatus::Running:
        return "running";
    case RuntimeTaskStatus::Completed:
        return "completed";
    case RuntimeTaskStatus::Failed:
        return "failed";
    }
    return "unknown";
}

namespace {

bool CleanupTempDirRoot(const std::filesystem::path& dir) {
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    return !ec;
}

bool TryPrepareTempDir(const std::filesystem::path& dir) {
    if (!CleanupTempDirRoot(dir)) return false;

    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    if (ec) return false;
    std::filesystem::permissions(dir, std::filesystem::perms::owner_all,
                                 std::filesystem::perm_options::replace, ec);
    return !ec;
}

bool SamePath(const std::filesystem::path& a, const std::filesystem::path& b) {
    return a.lexically_normal() == b.lexically_normal();
}

constexpr std::size_t kVectorSvgSpillThresholdBytes = 2 * 1024 * 1024;

SpillableArtifact SpillSvgToFile(const std::filesystem::path& path, const std::string& svg) {
    PendingArtifactFile pending(path);
    std::ofstream out(pending.path(), std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        throw std::runtime_error("failed to open svg spill file: " + path.string());
    }

    out.write(svg.data(), static_cast<std::streamsize>(svg.size()));
    out.close();

    std::error_code ec;
    const auto written_size = std::filesystem::file_size(path, ec);
    if (ec || written_size != svg.size()) {
        throw std::runtime_error("failed to persist full svg payload to spill file: " +
                                 path.string());
    }
    return pending.Commit(written_size);
}

} // namespace

TaskRuntime::TaskRuntime(std::int64_t worker_count, std::int64_t max_queue,
                         std::int64_t ttl_seconds, std::int64_t max_tasks_per_owner,
                         std::int64_t max_total_result_bytes, const std::string& data_dir)
    : max_queue_(max_queue), ttl_seconds_(ttl_seconds), max_tasks_per_owner_(max_tasks_per_owner),
      max_total_result_bytes_(max_total_result_bytes) {
    auto primary  = std::filesystem::path(data_dir) / "tmp" / "results";
    auto fallback = std::filesystem::temp_directory_path() / "chromaprint3d_results";

    if (!SamePath(primary, fallback) && !CleanupTempDirRoot(fallback)) {
        spdlog::warn("TaskRuntime: failed to clean stale fallback temp dir {}", fallback.string());
    }

    if (TryPrepareTempDir(primary)) {
        temp_dir_ = primary;
    } else {
        spdlog::warn("Cannot create temp dir {}, falling back to {}", primary.string(),
                     fallback.string());
        if (!TryPrepareTempDir(fallback)) {
            throw std::runtime_error(
                "Failed to create temp result directory at both " + primary.string() + " and " +
                fallback.string() +
                ". Check directory permissions (container user must own the tmpfs mount, "
                "e.g. tmpfs uid=10001,gid=10001 in docker-compose).");
        }
        temp_dir_ = fallback;
    }
    spdlog::info("TaskRuntime: temp result dir = {}", temp_dir_.string());

    StartWorkers(worker_count <= 0 ? 1 : worker_count);
    cleanup_thread_ = std::thread([this] { CleanupLoop(); });
}

TaskRuntime::~TaskRuntime() {
    shutdown_.store(true, std::memory_order_release);
    queue_cv_.notify_all();
    if (cleanup_thread_.joinable()) cleanup_thread_.join();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

SubmitResult TaskRuntime::SubmitConvertRaster(const std::string& owner,
                                              ChromaPrint3D::ConvertRasterRequest req,
                                              const std::string& input_name) {
    ConvertTaskPayload payload;
    payload.input_name = input_name;

    return SubmitInternal(owner, TaskKind::Convert, payload,
                          [this, request = std::move(req)](const std::string& id) mutable {
                              auto temp_path               = temp_dir_ / (id + "_model.3mf");
                              request.output_3mf_path      = temp_path.string();
                              request.output_3mf_file_only = true;
                              PendingArtifactFile pending_3mf(temp_path);

                              ChromaPrint3D::ProgressCallback progress_cb =
                                  [this, id](ChromaPrint3D::ConvertStage stage, float progress) {
                                      UpdateTaskPayload(id, [&](TaskPayload& p) {
                                          auto* cp = std::get_if<ConvertTaskPayload>(&p);
                                          if (!cp) return;
                                          cp->stage    = stage;
                                          cp->progress = progress;
                                      });
                                  };

                              auto result      = ChromaPrint3D::ConvertRaster(request, progress_cb);
                              auto file_size   = std::filesystem::file_size(temp_path);
                              auto spilled_3mf = pending_3mf.Commit(file_size);

                              UpdateTaskRecord(id, [&](TaskRecord& rec) {
                                  auto* cp = std::get_if<ConvertTaskPayload>(&rec.snapshot.payload);
                                  if (!cp) return;
                                  cp->result          = std::move(result);
                                  cp->progress        = 1.0f;
                                  cp->has_3mf_on_disk = true;
                                  rec.spilled_3mf     = std::move(spilled_3mf);
                              });
                          });
}

SubmitResult TaskRuntime::SubmitConvertVector(const std::string& owner,
                                              ChromaPrint3D::ConvertVectorRequest req,
                                              const std::string& input_name) {
    ConvertTaskPayload payload;
    payload.input_name = input_name;

    return SubmitInternal(owner, TaskKind::Convert, payload,
                          [this, request = std::move(req)](const std::string& id) mutable {
                              auto temp_path               = temp_dir_ / (id + "_model.3mf");
                              request.output_3mf_path      = temp_path.string();
                              request.output_3mf_file_only = true;
                              PendingArtifactFile pending_3mf(temp_path);

                              ChromaPrint3D::ProgressCallback progress_cb =
                                  [this, id](ChromaPrint3D::ConvertStage stage, float progress) {
                                      UpdateTaskPayload(id, [&](TaskPayload& p) {
                                          auto* cp = std::get_if<ConvertTaskPayload>(&p);
                                          if (!cp) return;
                                          cp->stage    = stage;
                                          cp->progress = progress;
                                      });
                                  };

                              auto result      = ChromaPrint3D::ConvertVector(request, progress_cb);
                              auto file_size   = std::filesystem::file_size(temp_path);
                              auto spilled_3mf = pending_3mf.Commit(file_size);

                              UpdateTaskRecord(id, [&](TaskRecord& rec) {
                                  auto* cp = std::get_if<ConvertTaskPayload>(&rec.snapshot.payload);
                                  if (!cp) return;
                                  cp->result          = std::move(result);
                                  cp->progress        = 1.0f;
                                  cp->has_3mf_on_disk = true;
                                  rec.spilled_3mf     = std::move(spilled_3mf);
                              });
                          });
}

SubmitResult TaskRuntime::SubmitConvertRasterMatchOnly(const std::string& owner,
                                                       ChromaPrint3D::ConvertRasterRequest req,
                                                       const std::string& input_name) {
    ConvertTaskPayload payload;
    payload.input_name = input_name;
    payload.match_only = true;

    return SubmitInternal(
        owner, TaskKind::Convert, payload,
        [this, request = std::move(req)](const std::string& id) mutable {
            if (request.dither != ChromaPrint3D::DitherMethod::None) {
                throw std::runtime_error("Recipe editing requires dither=None");
            }

            ChromaPrint3D::ProgressCallback progress_cb =
                [this, id](ChromaPrint3D::ConvertStage stage, float progress) {
                    UpdateTaskPayload(id, [&](TaskPayload& p) {
                        auto* cp = std::get_if<ConvertTaskPayload>(&p);
                        if (!cp) return;
                        cp->stage    = stage;
                        cp->progress = progress;
                    });
                };

            auto match_result = ChromaPrint3D::MatchRaster(request, progress_cb);

            std::set<std::string> unique_recipes;
            const int cl        = match_result.recipe_map.color_layers;
            const auto& recipes = match_result.recipe_map.recipes;
            const auto& mask    = match_result.recipe_map.mask;
            const int total_px  = match_result.recipe_map.width * match_result.recipe_map.height;
            for (int i = 0; i < total_px; ++i) {
                if (mask[static_cast<std::size_t>(i)] == 0) continue;
                std::string key;
                for (int l = 0; l < cl; ++l) {
                    if (l > 0) key += '-';
                    key += std::to_string(
                        recipes[static_cast<std::size_t>(i) * static_cast<std::size_t>(cl) +
                                static_cast<std::size_t>(l)]);
                }
                unique_recipes.insert(std::move(key));
            }

            constexpr int kMaxUniqueRecipes = 128;
            if (static_cast<int>(unique_recipes.size()) > kMaxUniqueRecipes) {
                throw std::runtime_error(
                    "Too many unique recipes: " + std::to_string(unique_recipes.size()) + ", max " +
                    std::to_string(kMaxUniqueRecipes));
            }

            auto region_map    = ChromaPrint3D::RasterRegionMap::Build(match_result.recipe_map);
            auto region_binary = region_map.SerializeRegionIds();

            cv::Mat preview_bgra = match_result.recipe_map.ToBgraImage();
            std::vector<uint8_t> preview_png;
            if (!preview_bgra.empty()) { preview_png = ChromaPrint3D::EncodePng(preview_bgra); }

            cv::Mat source_mask_img = match_result.recipe_map.ToSourceMaskImage();
            std::vector<uint8_t> source_mask_png;
            if (!source_mask_img.empty()) {
                source_mask_png = ChromaPrint3D::EncodePng(source_mask_img);
            }

            UpdateTaskRecord(id, [&](TaskRecord& rec) {
                auto* cp = std::get_if<ConvertTaskPayload>(&rec.snapshot.payload);
                if (!cp) return;
                cp->result.stats             = match_result.stats;
                cp->result.input_width       = match_result.input_width;
                cp->result.input_height      = match_result.input_height;
                cp->result.resolved_pixel_mm = match_result.resolved_pixel_mm;
                cp->result.physical_width_mm =
                    static_cast<float>(match_result.input_width) * match_result.resolved_pixel_mm;
                cp->result.physical_height_mm =
                    static_cast<float>(match_result.input_height) * match_result.resolved_pixel_mm;
                cp->result.preview_png     = std::move(preview_png);
                cp->result.source_mask_png = std::move(source_mask_png);
                cp->progress               = 1.0f;
                cp->match_only             = true;
                cp->raster_region_map      = std::move(region_map);
                cp->region_map_binary      = std::move(region_binary);
                cp->raster_match_state     = std::make_optional(std::move(match_result));
            });
        });
}

SubmitResult TaskRuntime::SubmitConvertVectorMatchOnly(const std::string& owner,
                                                       ChromaPrint3D::ConvertVectorRequest req,
                                                       const std::string& input_name) {
    ConvertTaskPayload payload;
    payload.input_name = input_name;
    payload.match_only = true;

    return SubmitInternal(
        owner, TaskKind::Convert, payload,
        [this, request = std::move(req)](const std::string& id) mutable {
            ChromaPrint3D::ProgressCallback progress_cb =
                [this, id](ChromaPrint3D::ConvertStage stage, float progress) {
                    UpdateTaskPayload(id, [&](TaskPayload& p) {
                        auto* cp = std::get_if<ConvertTaskPayload>(&p);
                        if (!cp) return;
                        cp->stage    = stage;
                        cp->progress = progress;
                    });
                };

            auto match_result = ChromaPrint3D::MatchVector(request, progress_cb);

            std::set<std::string> unique_recipes;
            for (const auto& entry : match_result.recipe_map.entries) {
                std::string key;
                for (std::size_t l = 0; l < entry.recipe.size(); ++l) {
                    if (l > 0) key += '-';
                    key += std::to_string(entry.recipe[l]);
                }
                unique_recipes.insert(std::move(key));
            }

            constexpr int kMaxUniqueRecipes = 128;
            if (static_cast<int>(unique_recipes.size()) > kMaxUniqueRecipes) {
                throw std::runtime_error(
                    "Too many unique recipes: " + std::to_string(unique_recipes.size()) + ", max " +
                    std::to_string(kMaxUniqueRecipes));
            }

            cv::Mat region_ids_mat  = ChromaPrint3D::RenderVectorRegionIds(match_result.proc_result,
                                                                           match_result.recipe_map);
            const std::size_t total = static_cast<std::size_t>(region_ids_mat.rows) *
                                      static_cast<std::size_t>(region_ids_mat.cols);
            std::vector<uint8_t> region_binary(total * sizeof(uint32_t));
            std::memcpy(region_binary.data(), region_ids_mat.data, region_binary.size());

            auto preview_png = ChromaPrint3D::RenderVectorPreviewPng(
                match_result.proc_result, match_result.recipe_map, match_result.profile.palette);
            auto source_mask_png = ChromaPrint3D::RenderVectorSourceMaskPng(
                match_result.proc_result, match_result.recipe_map);

            const int img_w = region_ids_mat.cols;
            const int img_h = region_ids_mat.rows;

            UpdateTaskRecord(id, [&](TaskRecord& rec) {
                auto* cp = std::get_if<ConvertTaskPayload>(&rec.snapshot.payload);
                if (!cp) return;
                cp->result.stats              = match_result.stats;
                cp->result.input_width        = img_w;
                cp->result.input_height       = img_h;
                cp->result.physical_width_mm  = match_result.proc_result.width_mm;
                cp->result.physical_height_mm = match_result.proc_result.height_mm;
                cp->result.preview_png        = std::move(preview_png);
                cp->result.source_mask_png    = std::move(source_mask_png);
                cp->progress                  = 1.0f;
                cp->match_only                = true;
                cp->region_map_binary         = std::move(region_binary);
                cp->vector_match_state        = std::make_optional(std::move(match_result));
            });
        });
}

SubmitResult TaskRuntime::SubmitMatting(const std::string& owner, std::vector<uint8_t> image_buffer,
                                        const std::string& method, const std::string& image_name,
                                        const ChromaPrint3D::MattingRegistry& registry) {
    MattingTaskPayload payload;
    payload.method     = method;
    payload.image_name = image_name;
    auto provider      = registry.GetShared(method);
    if (!provider) { return {false, 400, "", "Unknown matting method: " + method}; }

    return SubmitInternal(
        owner, TaskKind::Matting, payload,
        [this, buf = std::move(image_buffer),
         provider = std::move(provider)](const std::string& id) mutable {
            using Clock = std::chrono::steady_clock;
            auto start  = Clock::now();

            auto t0       = Clock::now();
            cv::Mat input = ChromaPrint3D::DecodeImageWithIccBgr(buf);
            auto t1       = Clock::now();
            if (input.empty()) throw std::runtime_error("Failed to decode image");

            ChromaPrint3D::MattingTimingInfo timing;
            auto output = provider->Run(input, &timing);
            if (output.mask.empty()) throw std::runtime_error("Matting produced empty mask");
            if (output.mask.size() != input.size()) {
                cv::resize(output.mask, output.mask, input.size(), 0, 0, cv::INTER_NEAREST);
            }
            if (!output.alpha.empty() && output.alpha.size() != input.size()) {
                cv::resize(output.alpha, output.alpha, input.size(), 0, 0, cv::INTER_LINEAR);
            }
            cv::Mat foreground = CompositeForeground(input, output.mask);

            auto t2 = Clock::now();
            std::vector<uint8_t> mask_png;
            std::vector<uint8_t> fg_png;
            cv::imencode(".png", output.mask, mask_png);
            cv::imencode(".png", foreground, fg_png);

            std::vector<uint8_t> alpha_png;
            std::vector<uint8_t> original_png;
            bool has_alpha = !output.alpha.empty();
            if (has_alpha) { cv::imencode(".png", output.alpha, alpha_png); }
            cv::imencode(".png", input, original_png);
            auto t3 = Clock::now();

            auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };

            UpdateTaskPayload(id, [&](TaskPayload& p) {
                auto* mp = std::get_if<MattingTaskPayload>(&p);
                if (!mp) return;
                mp->timing         = timing;
                mp->decode_ms      = ms(t1 - t0);
                mp->encode_ms      = ms(t3 - t2);
                mp->pipeline_ms    = ms(t3 - start);
                mp->mask_png       = std::move(mask_png);
                mp->foreground_png = std::move(fg_png);
                mp->width          = input.cols;
                mp->height         = input.rows;
                mp->alpha_png      = std::move(alpha_png);
                mp->original_png   = std::move(original_png);
                mp->has_alpha      = has_alpha;
            });
        });
}

SubmitResult TaskRuntime::SubmitVectorize(const std::string& owner,
                                          std::vector<uint8_t> image_buffer,
                                          neroued::vectorizer::VectorizerConfig config,
                                          const std::string& image_name) {
    VectorizeTaskPayload payload;
    payload.image_name = image_name;
    payload.config     = config;

    return SubmitInternal(
        owner, TaskKind::Vectorize, payload,
        [this, buf = std::move(image_buffer), config](const std::string& id) mutable {
            using Clock                   = std::chrono::steady_clock;
            auto t0                       = Clock::now();
            const std::size_t input_bytes = buf.size();
            spdlog::info(
                "Vectorize task core start: task_id={}, image_bytes={}, num_colors={}, "
                "min_region_area={}, curve_fit_error={:.3f}, corner_angle_threshold={:.1f}, "
                "smoothing_spatial={:.1f}, smoothing_color={:.1f}, upscale_short_edge={}, "
                "max_working_pixels={}, slic_region_size={}, svg_enable_stroke={}, "
                "svg_stroke_width={:.2f}",
                id, input_bytes, config.num_colors, config.min_region_area, config.curve_fit_error,
                config.corner_angle_threshold, config.smoothing_spatial, config.smoothing_color,
                config.upscale_short_edge, config.max_working_pixels, config.slic_region_size,
                config.svg_enable_stroke, config.svg_stroke_width);
            neroued::vectorizer::VectorizerResult result;
            try {
                result = neroued::vectorizer::Vectorize(buf.data(), buf.size(), config);
            } catch (const std::exception& e) {
                spdlog::error("Vectorize task core failed: task_id={}, image_bytes={}, error={}",
                              id, input_bytes, e.what());
                throw;
            } catch (...) {
                spdlog::error(
                    "Vectorize task core failed: task_id={}, image_bytes={}, unknown error", id,
                    input_bytes);
                throw;
            }
            auto t1                       = Clock::now();
            const std::size_t svg_bytes   = result.svg_content.size();
            const int out_width           = result.width;
            const int out_height          = result.height;
            const int out_shapes          = result.num_shapes;
            const int resolved_num_colors = result.resolved_num_colors;
            std::string svg_content       = std::move(result.svg_content);

            std::optional<SpillableArtifact> spilled_svg;
            if (svg_bytes >= kVectorSvgSpillThresholdBytes) {
                auto spill_path = temp_dir_ / (id + "_vectorized.svg");
                spilled_svg     = SpillSvgToFile(spill_path, svg_content);
            }
            const bool has_svg_on_disk = spilled_svg.has_value();

            auto ms = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };

            UpdateTaskRecord(id, [&, svg_bytes, svg_content = std::move(svg_content),
                                  spilled_svg = std::move(spilled_svg)](TaskRecord& rec) mutable {
                auto* vp = std::get_if<VectorizeTaskPayload>(&rec.snapshot.payload);
                if (!vp) return;
                vp->decode_ms           = 0;
                vp->vectorize_ms        = ms(t1 - t0);
                vp->pipeline_ms         = ms(t1 - t0);
                vp->svg_bytes           = svg_bytes;
                vp->width               = out_width;
                vp->height              = out_height;
                vp->num_shapes          = out_shapes;
                vp->resolved_num_colors = resolved_num_colors;
                if (spilled_svg.has_value()) {
                    vp->svg_content.clear();
                    vp->has_svg_on_disk = true;
                    rec.spilled_svg     = std::move(*spilled_svg);
                } else {
                    vp->svg_content     = std::move(svg_content);
                    vp->has_svg_on_disk = false;
                }
            });
            spdlog::info(
                "Vectorize task core completed: task_id={}, vectorize_ms={:.2f}, width={}, "
                "height={}, num_shapes={}, svg_bytes={}, spilled_svg={}",
                id, ms(t1 - t0), out_width, out_height, out_shapes, svg_bytes, has_svg_on_disk);
        });
}

SubmitResult TaskRuntime::CanAccept(const std::string& owner) const {
    if (owner.empty()) { return {false, 401, "", "Session is required"}; }
    std::scoped_lock lock(queue_mtx_, task_mtx_);
    if (static_cast<std::int64_t>(queue_.size()) >= max_queue_) {
        spdlog::warn("TaskRuntime gate reject: owner={}, reason=queue_full, queue_size={}, max={}",
                     OwnerTag(owner), queue_.size(), max_queue_);
        return {false, 429, "", "Task queue is full"};
    }
    const std::size_t owner_active = ActiveTasksByOwnerLocked(owner);
    if (static_cast<std::int64_t>(owner_active) >= max_tasks_per_owner_) {
        spdlog::warn(
            "TaskRuntime gate reject: owner={}, reason=too_many_active_tasks, active={}, max={}",
            OwnerTag(owner), owner_active, max_tasks_per_owner_);
        return {false, 429, "", "Too many active tasks for current session"};
    }
    return {true, 202, "", ""};
}

bool TaskRuntime::PostprocessMatting(const std::string& owner, const std::string& id,
                                     const ChromaPrint3D::MattingPostprocessParams& params,
                                     int& status_code, std::string& message) {
    std::lock_guard<std::mutex> lock(task_mtx_);
    auto it = tasks_.find(id);
    if (it == tasks_.end() || it->second.snapshot.owner != owner) {
        status_code = 404;
        message     = "Task not found";
        return false;
    }
    auto& snap = it->second.snapshot;
    if (snap.status != RuntimeTaskStatus::Completed) {
        status_code = 409;
        message     = "Task not completed";
        return false;
    }
    auto* mp = std::get_if<MattingTaskPayload>(&snap.payload);
    if (!mp) {
        status_code = 400;
        message     = "Task is not a matting task";
        return false;
    }
    if (mp->original_png.empty()) {
        status_code = 409;
        message     = "Original image not available";
        return false;
    }

    cv::Mat alpha;
    if (!mp->alpha_png.empty()) { alpha = cv::imdecode(mp->alpha_png, cv::IMREAD_GRAYSCALE); }
    cv::Mat fallback_mask = cv::imdecode(mp->mask_png, cv::IMREAD_GRAYSCALE);
    cv::Mat original      = cv::imdecode(mp->original_png, cv::IMREAD_COLOR);
    if (original.empty()) {
        status_code = 500;
        message     = "Failed to decode original image";
        return false;
    }

    auto result = ChromaPrint3D::ApplyMattingPostprocess(alpha, fallback_mask, original, params);

    std::size_t old_bytes = it->second.artifact_bytes;

    cv::imencode(".png", result.mask, mp->processed_mask_png);
    if (!result.foreground.empty()) {
        cv::imencode(".png", result.foreground, mp->processed_fg_png);
    } else {
        mp->processed_fg_png.clear();
    }
    if (!result.outline.empty()) {
        cv::imencode(".png", result.outline, mp->outline_png);
    } else {
        mp->outline_png.clear();
    }

    std::size_t new_bytes     = ComputeArtifactBytes(snap);
    total_artifact_bytes_     = total_artifact_bytes_ - old_bytes + new_bytes;
    it->second.artifact_bytes = new_bytes;

    status_code = 200;
    return true;
}

// --------------- Recipe Editor Sync Methods ---------------

std::optional<nlohmann::json> TaskRuntime::GetRecipeEditorSummary(const std::string& owner,
                                                                  const std::string& id) const {
    std::lock_guard<std::mutex> lock(task_mtx_);
    auto it = tasks_.find(id);
    if (it == tasks_.end() || it->second.snapshot.owner != owner) return std::nullopt;
    if (it->second.snapshot.status != RuntimeTaskStatus::Completed) return std::nullopt;
    return GetRecipeEditorSummaryLocked(it->second);
}

std::optional<nlohmann::json>
TaskRuntime::QueryRecipeAlternatives(const std::string& owner, const std::string& id,
                                     const ChromaPrint3D::Lab& target_lab, int max_candidates,
                                     int offset, const ChromaPrint3D::ModelPackage* model_pack) {
    std::lock_guard<std::mutex> lock(task_mtx_);
    auto it = tasks_.find(id);
    if (it == tasks_.end() || it->second.snapshot.owner != owner) return std::nullopt;

    auto* cp = std::get_if<ConvertTaskPayload>(&it->second.snapshot.payload);
    if (!cp || !cp->match_only) return std::nullopt;
    if (it->second.snapshot.status != RuntimeTaskStatus::Completed) return std::nullopt;

    const bool is_raster = cp->raster_match_state.has_value();
    const bool is_vector = cp->vector_match_state.has_value();
    if (!is_raster && !is_vector) return std::nullopt;

    if (!cp->recipe_search_cache.IsValid()) {
        const auto& dbs = is_raster ? cp->raster_match_state->dbs : cp->vector_match_state->dbs;
        const auto& profile =
            is_raster ? cp->raster_match_state->profile : cp->vector_match_state->profile;
        const auto& match_config =
            is_raster ? cp->raster_match_state->match_config : cp->vector_match_state->match_config;
        const auto& model_gate_c =
            is_raster ? cp->raster_match_state->model_gate : cp->vector_match_state->model_gate;
        cp->recipe_search_cache = ChromaPrint3D::RecipeSearchCache::Build(
            dbs, profile, match_config, model_pack, model_gate_c);
    }

    auto candidates = ChromaPrint3D::FindAlternativeRecipes(target_lab, cp->recipe_search_cache,
                                                            max_candidates, offset);

    nlohmann::json arr = nlohmann::json::array();
    for (const auto& c : candidates) {
        std::string hex = LabToHex(c.predicted_color);

        nlohmann::json obj = {{"recipe", c.recipe},
                              {"predicted_lab",
                               {{"L", c.predicted_color.l()},
                                {"a", c.predicted_color.a()},
                                {"b", c.predicted_color.b()}}},
                              {"hex", hex},
                              {"delta_e76", c.delta_e76},
                              {"lightness_diff", c.lightness_diff},
                              {"hue_diff", c.hue_diff},
                              {"from_model", c.from_model}};
        arr.push_back(std::move(obj));
    }
    return arr;
}

bool TaskRuntime::ReplaceRecipe(const std::string& owner, const std::string& id,
                                const std::vector<int>& target_region_ids,
                                const std::vector<uint8_t>& new_recipe,
                                const ChromaPrint3D::Lab& new_mapped_color, bool new_from_model,
                                int& status_code, std::string& message,
                                nlohmann::json& out_summary) {
    std::lock_guard<std::mutex> lock(task_mtx_);
    auto it = tasks_.find(id);
    if (it == tasks_.end() || it->second.snapshot.owner != owner) {
        status_code = 404;
        message     = "task not found";
        return false;
    }
    if (it->second.snapshot.status != RuntimeTaskStatus::Completed) {
        status_code = 409;
        message     = "task not in completed state";
        return false;
    }

    auto* cp = std::get_if<ConvertTaskPayload>(&it->second.snapshot.payload);
    if (!cp || !cp->match_only) {
        status_code = 400;
        message     = "not a match_only task";
        return false;
    }

    const bool is_raster = cp->raster_match_state.has_value();
    const bool is_vector = cp->vector_match_state.has_value();
    if (!is_raster && !is_vector) {
        status_code = 500;
        message     = "missing match state";
        return false;
    }

    if (is_raster) {
        if (!cp->raster_region_map.has_value()) {
            status_code = 500;
            message     = "missing raster region map";
            return false;
        }

        auto& mstate = *cp->raster_match_state;
        auto& rmap   = *cp->raster_region_map;

        try {
            mstate.recipe_map.ReplaceRecipeInRegions(rmap, target_region_ids, new_recipe,
                                                     new_mapped_color, new_from_model);
        } catch (const std::exception& e) {
            status_code = 400;
            message     = e.what();
            return false;
        }

        for (int rid : target_region_ids) {
            if (rid < 0 || static_cast<std::size_t>(rid) >= rmap.regions.size()) continue;
            auto& info        = rmap.regions[static_cast<std::size_t>(rid)];
            info.recipe       = new_recipe;
            info.mapped_color = new_mapped_color;
            info.from_model   = new_from_model;
        }

        cv::Mat preview_bgra = mstate.recipe_map.ToBgraImage();
        if (!preview_bgra.empty()) {
            cp->result.preview_png = ChromaPrint3D::EncodePng(preview_bgra);
        }
        cv::Mat source_mask_img = mstate.recipe_map.ToSourceMaskImage();
        if (!source_mask_img.empty()) {
            cp->result.source_mask_png = ChromaPrint3D::EncodePng(source_mask_img);
        }
    } else {
        auto& vstate = *cp->vector_match_state;

        try {
            vstate.recipe_map.ReplaceRecipeForEntries(target_region_ids, new_recipe,
                                                      new_mapped_color, new_from_model);
        } catch (const std::exception& e) {
            status_code = 400;
            message     = e.what();
            return false;
        }

        cp->result.preview_png = ChromaPrint3D::RenderVectorPreviewPng(
            vstate.proc_result, vstate.recipe_map, vstate.profile.palette);
        cp->result.source_mask_png =
            ChromaPrint3D::RenderVectorSourceMaskPng(vstate.proc_result, vstate.recipe_map);
    }

    std::size_t old_bytes     = it->second.artifact_bytes;
    std::size_t new_bytes     = ComputeArtifactBytes(it->second.snapshot);
    total_artifact_bytes_     = total_artifact_bytes_ - old_bytes + new_bytes;
    it->second.artifact_bytes = new_bytes;

    auto summary = GetRecipeEditorSummaryLocked(it->second);
    if (summary.has_value()) { out_summary = std::move(*summary); }

    status_code = 200;
    return true;
}

SubmitResult TaskRuntime::SubmitGenerateModel(const std::string& owner, const std::string& id,
                                              const ChromaPrint3D::ModelPackage* /*model_pack*/) {
    {
        std::lock_guard<std::mutex> lock(task_mtx_);
        auto it = tasks_.find(id);
        if (it == tasks_.end() || it->second.snapshot.owner != owner) {
            return {false, 404, id, "task not found"};
        }
        if (it->second.snapshot.status != RuntimeTaskStatus::Completed) {
            return {false, 409, id, "task not in completed state"};
        }
        auto* cp = std::get_if<ConvertTaskPayload>(&it->second.snapshot.payload);
        if (!cp || !cp->match_only) { return {false, 400, id, "not a match_only task"}; }
        if (!cp->raster_match_state.has_value() && !cp->vector_match_state.has_value()) {
            return {false, 500, id, "missing match state"};
        }

        cp->generate_error.clear();
        cp->result.model_3mf.clear();
        it->second.spilled_3mf           = SpillableArtifact{};
        it->second.snapshot.status       = RuntimeTaskStatus::Running;
        it->second.snapshot.completed_at = {};
        cp->stage                        = ChromaPrint3D::ConvertStage::BuildingModel;
        cp->progress                     = 0.0f;
    }

    RunTask(id, [this, id](const std::string& /*tid*/) {
        bool use_vector = false;
        {
            std::lock_guard<std::mutex> lock(task_mtx_);
            auto it    = tasks_.find(id);
            auto* cp   = std::get_if<ConvertTaskPayload>(&it->second.snapshot.payload);
            use_vector = cp->vector_match_state.has_value();
        }

        ChromaPrint3D::ProgressCallback progress_cb = [this, id](ChromaPrint3D::ConvertStage stage,
                                                                 float progress) {
            UpdateTaskPayload(id, [&](TaskPayload& p) {
                auto* cp = std::get_if<ConvertTaskPayload>(&p);
                if (!cp) return;
                cp->stage    = stage;
                cp->progress = progress;
            });
        };

        try {
            ChromaPrint3D::ConvertResult gen_result;

            if (use_vector) {
                ChromaPrint3D::MatchVectorResult match_copy;
                {
                    std::lock_guard<std::mutex> lock(task_mtx_);
                    auto it    = tasks_.find(id);
                    auto* cp   = std::get_if<ConvertTaskPayload>(&it->second.snapshot.payload);
                    match_copy = *cp->vector_match_state;
                }
                gen_result = ChromaPrint3D::GenerateVectorModel(match_copy, progress_cb);
            } else {
                ChromaPrint3D::MatchRasterResult match_copy;
                {
                    std::lock_guard<std::mutex> lock(task_mtx_);
                    auto it    = tasks_.find(id);
                    auto* cp   = std::get_if<ConvertTaskPayload>(&it->second.snapshot.payload);
                    match_copy = *cp->raster_match_state;
                }
                gen_result = ChromaPrint3D::GenerateRasterModel(match_copy, progress_cb);
            }

            PendingArtifactFile pending_3mf;
            SpillableArtifact spilled_3mf;
            if (!gen_result.model_3mf.empty()) {
                auto spill_path = temp_dir_ / (id + "_gen.3mf");
                pending_3mf     = PendingArtifactFile(spill_path);
                spdlog::info("GenerateModel: writing 3MF for task {}, {} bytes to {}", id,
                             gen_result.model_3mf.size(), spill_path.string());

                std::ofstream ofs(spill_path, std::ios::binary | std::ios::trunc);
                if (!ofs.is_open()) {
                    throw std::runtime_error("Cannot open 3MF spill file: " + spill_path.string());
                }
                ofs.write(reinterpret_cast<const char*>(gen_result.model_3mf.data()),
                          static_cast<std::streamsize>(gen_result.model_3mf.size()));
                if (!ofs.good()) {
                    throw std::runtime_error("Failed to write 3MF spill file: " +
                                             spill_path.string());
                }
                ofs.close();
                if (ofs.fail()) {
                    throw std::runtime_error("Failed to flush/close 3MF spill file: " +
                                             spill_path.string());
                }

                auto actual_size = std::filesystem::file_size(spill_path);
                if (actual_size != gen_result.model_3mf.size()) {
                    spdlog::error("GenerateModel: 3MF file size mismatch for task {}: expected={}, "
                                  "actual={}",
                                  id, gen_result.model_3mf.size(), actual_size);
                    throw std::runtime_error("3MF file size mismatch after write");
                }

                spilled_3mf = pending_3mf.Commit(actual_size);
                gen_result.model_3mf.clear();
                gen_result.model_3mf.shrink_to_fit();
                spdlog::info("GenerateModel: 3MF written for task {}, {} bytes", id, actual_size);
            }

            UpdateTaskRecord(id, [&](TaskRecord& rec) {
                auto* cp = std::get_if<ConvertTaskPayload>(&rec.snapshot.payload);
                if (!cp) return;
                cp->result.model_3mf       = std::move(gen_result.model_3mf);
                cp->result.layer_previews  = std::move(gen_result.layer_previews);
                cp->result.preview_png     = std::move(gen_result.preview_png);
                cp->result.source_mask_png = std::move(gen_result.source_mask_png);
                cp->progress               = 1.0f;
                cp->has_3mf_on_disk        = spilled_3mf.has_file();
                rec.spilled_3mf            = std::move(spilled_3mf);
            });
        } catch (const std::exception& e) {
            spdlog::error("GenerateModel failed for task {}: {}", id, e.what());
            UpdateTaskPayload(id, [&](TaskPayload& p) {
                auto* cp = std::get_if<ConvertTaskPayload>(&p);
                if (!cp) return;
                cp->generate_error = e.what();
                cp->result.model_3mf.clear();
            });
        }
    });

    return {true, 200, id, ""};
}

std::vector<TaskSnapshot> TaskRuntime::ListTasks(const std::string& owner) const {
    std::vector<TaskSnapshot> out;
    if (owner.empty()) return out;
    std::lock_guard<std::mutex> lock(task_mtx_);
    out.reserve(tasks_.size());
    for (const auto& [_, task] : tasks_) {
        if (task.snapshot.owner == owner) out.push_back(task.snapshot);
    }
    std::sort(out.begin(), out.end(), [](const TaskSnapshot& a, const TaskSnapshot& b) {
        return a.created_at > b.created_at;
    });
    return out;
}

std::optional<TaskSnapshot> TaskRuntime::FindTask(const std::string& owner,
                                                  const std::string& id) const {
    std::lock_guard<std::mutex> lock(task_mtx_);
    auto it = tasks_.find(id);
    if (it == tasks_.end()) return std::nullopt;
    if (it->second.snapshot.owner != owner) return std::nullopt;
    return it->second.snapshot;
}

bool TaskRuntime::DeleteTask(const std::string& owner, const std::string& id, int& status_code,
                             std::string& message) {
    std::scoped_lock lock(queue_mtx_, task_mtx_);
    auto it = tasks_.find(id);
    if (it == tasks_.end() || it->second.snapshot.owner != owner) {
        status_code = 404;
        message     = "Task not found";
        return false;
    }
    if (it->second.snapshot.status == RuntimeTaskStatus::Pending) {
        auto queue_it = std::find_if(queue_.begin(), queue_.end(), [&id](const QueueEntry& entry) {
            return entry.task_id == id;
        });
        if (queue_it == queue_.end()) {
            status_code = 409;
            message     = "Task is being dispatched";
            return false;
        }
        queue_.erase(queue_it);
        total_artifact_bytes_ -= std::min(total_artifact_bytes_, it->second.artifact_bytes);
        tasks_.erase(it);
        status_code = 200;
        return true;
    }
    if (it->second.snapshot.status == RuntimeTaskStatus::Running) {
        status_code = 409;
        message     = "Cannot delete running task";
        return false;
    }
    total_artifact_bytes_ -= std::min(total_artifact_bytes_, it->second.artifact_bytes);
    tasks_.erase(it);
    status_code = 200;
    return true;
}

std::optional<TaskArtifact> TaskRuntime::LoadArtifact(const std::string& owner,
                                                      const std::string& id,
                                                      const std::string& artifact, int& status_code,
                                                      std::string& message) const {
    std::lock_guard<std::mutex> lock(task_mtx_);
    auto it = tasks_.find(id);
    if (it == tasks_.end() || it->second.snapshot.owner != owner) {
        status_code = 404;
        message     = "Task not found";
        return std::nullopt;
    }
    const auto& t = it->second.snapshot;
    if (t.status != RuntimeTaskStatus::Completed) {
        status_code = 409;
        message     = "Task not completed";
        return std::nullopt;
    }

    const auto& rec = it->second;

    if (auto cp = std::get_if<ConvertTaskPayload>(&t.payload)) {
        if (artifact == "result") {
            std::string fname =
                (cp->input_name.empty() ? id.substr(0, 8) : cp->input_name) + ".3mf";
            const std::string ct = "application/vnd.ms-package.3dmanufacturing-3dmodel+xml";
            if (rec.spilled_3mf.has_file()) {
                const auto& fpath  = rec.spilled_3mf.path();
                auto expected_size = rec.spilled_3mf.file_size();
                std::error_code ec;
                bool exists      = std::filesystem::exists(fpath, ec);
                auto actual_size = exists ? std::filesystem::file_size(fpath, ec) : 0ULL;
                spdlog::info("LoadArtifact: serving file-based 3MF for task {}: path={}, "
                             "exists={}, actual_size={}, expected_size={}",
                             id, fpath.string(), exists, actual_size, expected_size);
                if (!exists) {
                    spdlog::error("LoadArtifact: spilled 3MF file missing for task {}: {}", id,
                                  fpath.string());
                    status_code = 404;
                    message     = "3MF file missing from disk";
                    return std::nullopt;
                }
                if (expected_size > 0 && actual_size != expected_size) {
                    spdlog::error("LoadArtifact: 3MF file size mismatch for task {}: "
                                  "expected={}, actual={}",
                                  id, expected_size, actual_size);
                    status_code = 500;
                    message     = "3MF file is incomplete or corrupt";
                    return std::nullopt;
                }
                return TaskArtifact{{}, fpath, ct, std::move(fname)};
            }
            if (cp->result.model_3mf.empty()) {
                status_code = 404;
                message     = "3MF result not available";
                return std::nullopt;
            }
            return TaskArtifact{cp->result.model_3mf, {}, ct, std::move(fname)};
        }
        if (artifact == "preview") {
            if (cp->result.preview_png.empty()) {
                status_code = 404;
                message     = "Preview not available";
                return std::nullopt;
            }
            return TaskArtifact{cp->result.preview_png, {}, "image/png", "preview.png"};
        }
        if (artifact == "source-mask") {
            if (cp->result.source_mask_png.empty()) {
                status_code = 404;
                message     = "Source mask not available";
                return std::nullopt;
            }
            return TaskArtifact{cp->result.source_mask_png, {}, "image/png", "source_mask.png"};
        }
        if (artifact == "region-map") {
            if (cp->region_map_binary.empty()) {
                status_code = 404;
                message     = "Region map not available";
                return std::nullopt;
            }
            return TaskArtifact{
                cp->region_map_binary, {}, "application/octet-stream", "region_map.bin"};
        }
        if (artifact.rfind(kLayerPreviewArtifactPrefix, 0) == 0) {
            const std::string index_text =
                artifact.substr(std::strlen(kLayerPreviewArtifactPrefix));
            if (index_text.empty() ||
                !std::all_of(index_text.begin(), index_text.end(),
                             [](unsigned char c) { return std::isdigit(c) != 0; })) {
                status_code = 404;
                message     = "Invalid layer preview artifact";
                return std::nullopt;
            }
            std::size_t idx = 0;
            try {
                idx = static_cast<std::size_t>(std::stoul(index_text));
            } catch (...) {
                status_code = 404;
                message     = "Invalid layer preview artifact";
                return std::nullopt;
            }
            if (idx >= cp->result.layer_previews.layer_pngs.size()) {
                status_code = 404;
                message     = "Layer preview not available";
                return std::nullopt;
            }
            const auto& png = cp->result.layer_previews.layer_pngs[idx];
            if (png.empty()) {
                status_code = 404;
                message     = "Layer preview is empty";
                return std::nullopt;
            }
            return TaskArtifact{
                png, {}, "image/png", "layer-preview-" + std::to_string(idx) + ".png"};
        }
    }

    if (auto mp = std::get_if<MattingTaskPayload>(&t.payload)) {
        if (artifact == "mask") { return TaskArtifact{mp->mask_png, {}, "image/png", "mask.png"}; }
        if (artifact == "foreground") {
            return TaskArtifact{mp->foreground_png, {}, "image/png", "foreground.png"};
        }
        if (artifact == "alpha") {
            if (mp->alpha_png.empty()) {
                status_code = 404;
                message     = "Alpha map not available for this matting method";
                return std::nullopt;
            }
            return TaskArtifact{mp->alpha_png, {}, "image/png", "alpha.png"};
        }
        if (artifact == "processed-mask") {
            if (mp->processed_mask_png.empty()) {
                status_code = 404;
                message     = "Run postprocess first";
                return std::nullopt;
            }
            return TaskArtifact{mp->processed_mask_png, {}, "image/png", "processed_mask.png"};
        }
        if (artifact == "processed-foreground") {
            if (mp->processed_fg_png.empty()) {
                status_code = 404;
                message     = "Run postprocess first";
                return std::nullopt;
            }
            return TaskArtifact{mp->processed_fg_png, {}, "image/png", "processed_foreground.png"};
        }
        if (artifact == "outline") {
            if (mp->outline_png.empty()) {
                status_code = 404;
                message     = "Outline not available (run postprocess with outline enabled)";
                return std::nullopt;
            }
            return TaskArtifact{mp->outline_png, {}, "image/png", "outline.png"};
        }
    }

    if (auto vp = std::get_if<VectorizeTaskPayload>(&t.payload)) {
        if (artifact == "svg") {
            std::string filename = vp->image_name.empty() ? "result.svg" : vp->image_name + ".svg";
            if (rec.spilled_svg.has_file()) {
                return TaskArtifact{
                    {}, rec.spilled_svg.path(), "image/svg+xml", std::move(filename)};
            }
            if (vp->svg_content.empty()) {
                status_code = 404;
                message     = "SVG result not available";
                return std::nullopt;
            }
            std::vector<uint8_t> svg(vp->svg_content.begin(), vp->svg_content.end());
            return TaskArtifact{std::move(svg), {}, "image/svg+xml", std::move(filename)};
        }
    }

    status_code = 404;
    message     = "Artifact not found";
    return std::nullopt;
}

int TaskRuntime::ActiveTaskCount() const { return running_count_.load(std::memory_order_relaxed); }

SubmitResult TaskRuntime::SubmitInternal(const std::string& owner, TaskKind kind,
                                         TaskPayload payload, WorkerFn worker) {
    if (owner.empty()) { return {false, 401, "", "Session is required"}; }
    const std::string id     = NewTaskId();
    std::size_t queue_before = 0;
    std::size_t queue_after  = 0;
    std::size_t owner_active = 0;

    {
        std::scoped_lock lock(queue_mtx_, task_mtx_);
        queue_before = queue_.size();
        if (static_cast<std::int64_t>(queue_.size()) >= max_queue_) {
            spdlog::warn("Task enqueue rejected: id={}, kind={}, owner={}, reason=queue_full, "
                         "queue_size={}, "
                         "max={}",
                         id, TaskKindToString(kind), OwnerTag(owner), queue_.size(), max_queue_);
            return {false, 429, "", "Task queue is full"};
        }
        owner_active = ActiveTasksByOwnerLocked(owner);
        if (static_cast<std::int64_t>(owner_active) >= max_tasks_per_owner_) {
            spdlog::warn(
                "Task enqueue rejected: id={}, kind={}, owner={}, reason=too_many_active_tasks, "
                "active={}, max={}",
                id, TaskKindToString(kind), OwnerTag(owner), owner_active, max_tasks_per_owner_);
            return {false, 429, "", "Too many active tasks for current session"};
        }
        TaskRecord rec;
        rec.snapshot.id         = id;
        rec.snapshot.owner      = owner;
        rec.snapshot.kind       = kind;
        rec.snapshot.status     = RuntimeTaskStatus::Pending;
        rec.snapshot.created_at = std::chrono::steady_clock::now();
        rec.snapshot.payload    = std::move(payload);
        tasks_[id]              = std::move(rec);

        queue_.push_back(QueueEntry{id, std::move(worker)});
        queue_after = queue_.size();
        total_submitted_.fetch_add(1, std::memory_order_relaxed);
    }
    if (kind == TaskKind::Vectorize) {
        spdlog::info("Task enqueued: id={}, kind={}, owner={}, queue_before={}, queue_after={}, "
                     "owner_active={}, running={}",
                     id, TaskKindToString(kind), OwnerTag(owner), queue_before, queue_after,
                     owner_active, running_count_.load(std::memory_order_relaxed));
    } else {
        spdlog::debug("Task enqueued: id={}, kind={}, owner={}, queue_before={}, queue_after={}, "
                      "owner_active={}, running={}",
                      id, TaskKindToString(kind), OwnerTag(owner), queue_before, queue_after,
                      owner_active, running_count_.load(std::memory_order_relaxed));
    }
    queue_cv_.notify_one();
    return {true, 202, id, ""};
}

void TaskRuntime::RunTask(const std::string& id, WorkerFn worker) {
    TaskKind kind      = TaskKind::Convert;
    std::string owner  = "none";
    double queue_wait  = 0.0;
    auto start_running = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(task_mtx_);
        auto it = tasks_.find(id);
        if (it != tasks_.end()) {
            kind       = it->second.snapshot.kind;
            owner      = OwnerTag(it->second.snapshot.owner);
            queue_wait = ElapsedMs(it->second.snapshot.created_at, start_running);
        }
    }
    if (kind == TaskKind::Vectorize) {
        spdlog::info("Task running: id={}, kind={}, owner={}, queue_wait_ms={:.2f}", id,
                     TaskKindToString(kind), owner, queue_wait);
    } else {
        spdlog::debug("Task running: id={}, kind={}, owner={}, queue_wait_ms={:.2f}", id,
                      TaskKindToString(kind), owner, queue_wait);
    }
    MarkRunning(id);
    running_count_.fetch_add(1, std::memory_order_relaxed);
    try {
        worker(id);
        MarkCompleted(id);
    } catch (const std::exception& e) { MarkFailed(id, e.what()); } catch (...) {
        MarkFailed(id, "Unknown error");
    }
    running_count_.fetch_sub(1, std::memory_order_relaxed);
}

void TaskRuntime::StartWorkers(std::int64_t worker_count) {
    workers_.reserve(static_cast<std::size_t>(worker_count));
    for (std::int64_t i = 0; i < worker_count; ++i) {
        workers_.emplace_back([this] { WorkerLoop(); });
    }
}

void TaskRuntime::WorkerLoop() {
    while (!shutdown_.load(std::memory_order_acquire)) {
        QueueEntry job;
        {
            std::unique_lock<std::mutex> lock(queue_mtx_);
            queue_cv_.wait(lock, [this] {
                return shutdown_.load(std::memory_order_acquire) || !queue_.empty();
            });
            if (shutdown_.load(std::memory_order_acquire) && queue_.empty()) return;
            job = std::move(queue_.front());
            queue_.pop_front();
        }
        RunTask(job.task_id, std::move(job.worker));
    }
}

void TaskRuntime::MarkRunning(const std::string& id) {
    std::lock_guard<std::mutex> lock(task_mtx_);
    auto it = tasks_.find(id);
    if (it == tasks_.end()) return;
    it->second.snapshot.status = RuntimeTaskStatus::Running;
    spdlog::debug("Task state -> running: id={}, kind={}", id,
                  TaskKindToString(it->second.snapshot.kind));
}

void TaskRuntime::MarkFailed(const std::string& id, const std::string& message) {
    std::lock_guard<std::mutex> lock(task_mtx_);
    auto it = tasks_.find(id);
    if (it == tasks_.end()) return;
    it->second.snapshot.status       = RuntimeTaskStatus::Failed;
    it->second.snapshot.error        = message;
    it->second.snapshot.completed_at = std::chrono::steady_clock::now();
    const double elapsed =
        ElapsedMs(it->second.snapshot.created_at, it->second.snapshot.completed_at);
    spdlog::error("Task failed: id={}, kind={}, owner={}, elapsed_ms={:.2f}, error={}", id,
                  TaskKindToString(it->second.snapshot.kind), OwnerTag(it->second.snapshot.owner),
                  elapsed, message);
}

void TaskRuntime::MarkCompleted(const std::string& id) {
    std::lock_guard<std::mutex> lock(task_mtx_);
    auto it = tasks_.find(id);
    if (it == tasks_.end()) return;
    it->second.snapshot.status       = RuntimeTaskStatus::Completed;
    it->second.snapshot.completed_at = std::chrono::steady_clock::now();
    it->second.artifact_bytes        = ComputeArtifactBytes(it->second.snapshot);
    total_artifact_bytes_ += it->second.artifact_bytes;
    const double elapsed =
        ElapsedMs(it->second.snapshot.created_at, it->second.snapshot.completed_at);

    if (auto* vp = std::get_if<VectorizeTaskPayload>(&it->second.snapshot.payload)) {
        const std::size_t svg_bytes = vp->svg_bytes > 0 ? vp->svg_bytes : vp->svg_content.size();
        spdlog::info(
            "Task completed: id={}, kind={}, owner={}, elapsed_ms={:.2f}, vectorize_ms={:.2f}, "
            "pipeline_ms={:.2f}, width={}, height={}, num_shapes={}, svg_bytes={}, spilled_svg={}, "
            "artifact_bytes={}",
            id, TaskKindToString(it->second.snapshot.kind), OwnerTag(it->second.snapshot.owner),
            elapsed, vp->vectorize_ms, vp->pipeline_ms, vp->width, vp->height, vp->num_shapes,
            svg_bytes, it->second.spilled_svg.has_file(), it->second.artifact_bytes);
    } else {
        spdlog::debug(
            "Task completed: id={}, kind={}, owner={}, elapsed_ms={:.2f}, artifact_bytes={}", id,
            TaskKindToString(it->second.snapshot.kind), OwnerTag(it->second.snapshot.owner),
            elapsed, it->second.artifact_bytes);
    }
    EnforceResultBudgetLocked();
}

std::size_t TaskRuntime::ComputeArtifactBytes(const TaskSnapshot& snapshot) const {
    if (auto cp = std::get_if<ConvertTaskPayload>(&snapshot.payload)) {
        std::size_t layer_preview_bytes = 0;
        for (const auto& png : cp->result.layer_previews.layer_pngs) {
            layer_preview_bytes += png.size();
        }
        std::size_t bytes = cp->result.model_3mf.size() + cp->result.preview_png.size() +
                            cp->result.source_mask_png.size() + layer_preview_bytes;
        bytes += cp->region_map_binary.size();
        if (cp->raster_match_state.has_value()) {
            bytes += cp->raster_match_state->EstimateBytes();
        }
        if (cp->raster_region_map.has_value()) { bytes += cp->raster_region_map->EstimateBytes(); }
        if (cp->vector_match_state.has_value()) {
            bytes += cp->vector_match_state->EstimateBytes();
        }
        return bytes;
    }
    if (auto mp = std::get_if<MattingTaskPayload>(&snapshot.payload)) {
        return mp->mask_png.size() + mp->foreground_png.size() + mp->alpha_png.size() +
               mp->original_png.size() + mp->processed_mask_png.size() +
               mp->processed_fg_png.size() + mp->outline_png.size();
    }
    if (auto vp = std::get_if<VectorizeTaskPayload>(&snapshot.payload)) {
        return vp->svg_content.size();
    }
    return 0;
}

std::optional<nlohmann::json>
TaskRuntime::GetRecipeEditorSummaryLocked(const TaskRecord& rec) const {
    const auto* cp = std::get_if<ConvertTaskPayload>(&rec.snapshot.payload);
    if (!cp || !cp->match_only) return std::nullopt;

    const bool is_raster = cp->raster_match_state.has_value();
    const bool is_vector = cp->vector_match_state.has_value();
    if (!is_raster && !is_vector) return std::nullopt;

    if (is_raster) {
        if (!cp->raster_region_map.has_value()) return std::nullopt;

        const auto& rmap   = *cp->raster_region_map;
        const auto& mstate = *cp->raster_match_state;

        nlohmann::json regions_arr = nlohmann::json::array();
        std::vector<int> region_recipe_indices;
        region_recipe_indices.reserve(rmap.regions.size());

        std::map<std::string, int> recipe_to_idx;
        nlohmann::json unique_recipes_arr = nlohmann::json::array();

        for (const auto& reg : rmap.regions) {
            std::string key;
            for (std::size_t l = 0; l < reg.recipe.size(); ++l) {
                if (l > 0) key += '-';
                key += std::to_string(reg.recipe[l]);
            }

            int recipe_idx;
            auto rit = recipe_to_idx.find(key);
            if (rit == recipe_to_idx.end()) {
                recipe_idx         = static_cast<int>(unique_recipes_arr.size());
                recipe_to_idx[key] = recipe_idx;

                nlohmann::json recipe_obj = {{"recipe", reg.recipe},
                                             {"mapped_lab",
                                              {{"L", reg.mapped_color.l()},
                                               {"a", reg.mapped_color.a()},
                                               {"b", reg.mapped_color.b()}}},
                                             {"hex", LabToHex(reg.mapped_color)},
                                             {"from_model", reg.from_model}};
                unique_recipes_arr.push_back(std::move(recipe_obj));
            } else {
                recipe_idx = rit->second;
            }

            region_recipe_indices.push_back(recipe_idx);
            nlohmann::json reg_obj = {{"region_id", reg.region_id},
                                      {"recipe_index", recipe_idx},
                                      {"pixel_count", reg.pixel_count}};
            regions_arr.push_back(std::move(reg_obj));
        }

        nlohmann::json palette_arr = nlohmann::json::array();
        for (const auto& ch : mstate.profile.palette) {
            palette_arr.push_back(
                {{"color", ch.color}, {"material", ch.material}, {"hex_color", ch.hex_color}});
        }

        return nlohmann::json{{"width", rmap.width},
                              {"height", rmap.height},
                              {"region_count", static_cast<int>(rmap.regions.size())},
                              {"regions", std::move(regions_arr)},
                              {"unique_recipes", std::move(unique_recipes_arr)},
                              {"region_recipe_indices", region_recipe_indices},
                              {"palette", std::move(palette_arr)},
                              {"color_layers", mstate.recipe_map.color_layers}};
    }

    // vector path
    const auto& vstate  = *cp->vector_match_state;
    const auto& entries = vstate.recipe_map.entries;

    const auto& region_bin = cp->region_map_binary;
    const int img_w        = cp->result.input_width;
    const int img_h        = cp->result.input_height;

    std::vector<int> pixel_counts(entries.size(), 0);
    if (!region_bin.empty() && img_w > 0 && img_h > 0) {
        const auto* ids         = reinterpret_cast<const uint32_t*>(region_bin.data());
        const std::size_t total = static_cast<std::size_t>(img_w) * static_cast<std::size_t>(img_h);
        for (std::size_t i = 0; i < total; ++i) {
            uint32_t rid = ids[i];
            if (rid != 0xFFFFFFFF && rid < static_cast<uint32_t>(entries.size())) {
                ++pixel_counts[rid];
            }
        }
    }

    nlohmann::json regions_arr = nlohmann::json::array();
    std::vector<int> region_recipe_indices;
    region_recipe_indices.reserve(entries.size());

    std::map<std::string, int> recipe_to_idx;
    nlohmann::json unique_recipes_arr = nlohmann::json::array();

    for (std::size_t ei = 0; ei < entries.size(); ++ei) {
        const auto& entry = entries[ei];
        std::string key;
        for (std::size_t l = 0; l < entry.recipe.size(); ++l) {
            if (l > 0) key += '-';
            key += std::to_string(entry.recipe[l]);
        }

        int recipe_idx;
        auto rit = recipe_to_idx.find(key);
        if (rit == recipe_to_idx.end()) {
            recipe_idx         = static_cast<int>(unique_recipes_arr.size());
            recipe_to_idx[key] = recipe_idx;

            nlohmann::json recipe_obj = {{"recipe", entry.recipe},
                                         {"mapped_lab",
                                          {{"L", entry.matched_color.l()},
                                           {"a", entry.matched_color.a()},
                                           {"b", entry.matched_color.b()}}},
                                         {"hex", LabToHex(entry.matched_color)},
                                         {"from_model", entry.from_model}};
            unique_recipes_arr.push_back(std::move(recipe_obj));
        } else {
            recipe_idx = rit->second;
        }

        region_recipe_indices.push_back(recipe_idx);
        nlohmann::json reg_obj = {{"region_id", static_cast<int>(ei)},
                                  {"recipe_index", recipe_idx},
                                  {"pixel_count", pixel_counts[ei]}};
        regions_arr.push_back(std::move(reg_obj));
    }

    nlohmann::json palette_arr = nlohmann::json::array();
    for (const auto& ch : vstate.profile.palette) {
        palette_arr.push_back(
            {{"color", ch.color}, {"material", ch.material}, {"hex_color", ch.hex_color}});
    }

    return nlohmann::json{{"width", img_w},
                          {"height", img_h},
                          {"region_count", static_cast<int>(entries.size())},
                          {"regions", std::move(regions_arr)},
                          {"unique_recipes", std::move(unique_recipes_arr)},
                          {"region_recipe_indices", region_recipe_indices},
                          {"palette", std::move(palette_arr)},
                          {"color_layers", vstate.recipe_map.color_layers}};
}

void TaskRuntime::CleanupLoop() {
    while (!shutdown_.load(std::memory_order_acquire)) {
        std::this_thread::sleep_for(std::chrono::seconds(30));
        auto now = std::chrono::steady_clock::now();
        std::lock_guard<std::mutex> lock(task_mtx_);
        CleanupExpiredLocked(now);
    }
}

void TaskRuntime::CleanupExpiredLocked(const std::chrono::steady_clock::time_point& now) {
    for (auto it = tasks_.begin(); it != tasks_.end();) {
        const auto& t = it->second.snapshot;
        if (t.status == RuntimeTaskStatus::Completed || t.status == RuntimeTaskStatus::Failed) {
            auto elapsed =
                std::chrono::duration_cast<std::chrono::seconds>(now - t.completed_at).count();
            if (elapsed > ttl_seconds_) {
                total_artifact_bytes_ -= std::min(total_artifact_bytes_, it->second.artifact_bytes);
                it = tasks_.erase(it);
                continue;
            }
        }
        ++it;
    }
    EnforceResultBudgetLocked();
}

void TaskRuntime::EnforceResultBudgetLocked() {
    if (max_total_result_bytes_ <= 0) return;
    while (static_cast<std::int64_t>(total_artifact_bytes_) > max_total_result_bytes_) {
        auto oldest = tasks_.end();
        for (auto it = tasks_.begin(); it != tasks_.end(); ++it) {
            auto st = it->second.snapshot.status;
            if (st != RuntimeTaskStatus::Completed && st != RuntimeTaskStatus::Failed) continue;
            if (oldest == tasks_.end() ||
                it->second.snapshot.completed_at < oldest->second.snapshot.completed_at) {
                oldest = it;
            }
        }
        if (oldest == tasks_.end()) break;
        total_artifact_bytes_ -= std::min(total_artifact_bytes_, oldest->second.artifact_bytes);
        tasks_.erase(oldest);
    }
}

std::size_t TaskRuntime::ActiveTasksByOwnerLocked(const std::string& owner) const {
    std::size_t count = 0;
    for (const auto& [_, task] : tasks_) {
        if (task.snapshot.owner != owner) continue;
        if (task.snapshot.status == RuntimeTaskStatus::Pending ||
            task.snapshot.status == RuntimeTaskStatus::Running) {
            ++count;
        }
    }
    return count;
}

std::string TaskRuntime::NewTaskId() { return detail::RandomHex(16); }

} // namespace chromaprint3d::backend
