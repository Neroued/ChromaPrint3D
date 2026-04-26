#pragma once

/// \file matting.h
/// \brief Image matting (foreground extraction) via pluggable providers.

#include "export.h"

#include <opencv2/core/mat.hpp>

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ChromaPrint3D {

// ── Model configuration (loaded from JSON sidecar) ─────────────────────────

/// Preprocessing and postprocessing parameters for a matting model.
/// Loaded from a JSON file that sits beside the model file.
struct CHROMAPRINT3D_API MattingModelConfig {
    std::string name;
    std::string description;

    int input_width  = 320;
    int input_height = 320;

    std::string channel_order = "rgb";
    std::string layout        = "nchw";

    float normalize_scale               = 1.0f / 255.0f;
    std::array<float, 3> normalize_mean = {0.0f, 0.0f, 0.0f};
    std::array<float, 3> normalize_std  = {1.0f, 1.0f, 1.0f};

    int output_index = 0;
    float threshold  = 0.5f;

    /// Load config from a JSON file.
    static MattingModelConfig LoadFromJson(const std::string& json_path);
};

// ── Timing info ─────────────────────────────────────────────────────────────

/// Structured timing breakdown for a single matting invocation.
struct MattingTimingInfo {
    double preprocess_ms  = 0;
    double inference_ms   = 0;
    double postprocess_ms = 0;
    double total_ms       = 0;
};

// ── Provider output ─────────────────────────────────────────────────────────

/// Result of a matting provider invocation.
struct MattingOutput {
    cv::Mat mask;  ///< CV_8UC1, binary (0 = background, 255 = foreground).
    cv::Mat alpha; ///< CV_8UC1, continuous alpha [0, 255]. Empty when unavailable.
};

// ── Provider interface ──────────────────────────────────────────────────────

/// Abstract interface for a matting algorithm.
/// Implementations receive a BGR uint8 image and return a MattingOutput
/// containing at least a binary mask, and optionally a continuous alpha map.
class CHROMAPRINT3D_API IMattingProvider {
public:
    virtual ~IMattingProvider() = default;

    /// Human-readable name of this provider.
    virtual std::string Name() const = 0;

    /// Short description of what this provider does.
    virtual std::string Description() const = 0;

    /// Generate a foreground mask (and optional alpha) from a BGR image.
    /// If \p timing is non-null, it will be populated with per-stage timings.
    virtual MattingOutput Run(const cv::Mat& bgr, MattingTimingInfo* timing = nullptr) const = 0;
};

using MattingProviderPtr = std::shared_ptr<IMattingProvider>;

// ── Built-in OpenCV provider ────────────────────────────────────────────────

/// Matting based on background-color thresholding in Lab space.
/// Samples the four corners of the image as background reference, then marks
/// any pixel whose Lab distance exceeds the threshold as foreground.
class CHROMAPRINT3D_API ThresholdMattingProvider : public IMattingProvider {
public:
    /// \param distance_threshold ΔE76 distance to background; pixels above this
    ///        are foreground. Default `6.0f` matches project D65 Lab range
    ///        (L in [0,100]); the previous default `15.0f` targeted OpenCV
    ///        uint8 Lab range (L in [0,255]). Divide an OpenCV-tuned value by
    ///        ~2.55 when porting calibration to project D65.
    explicit ThresholdMattingProvider(float distance_threshold = 6.0f);

    std::string Name() const override;
    std::string Description() const override;
    MattingOutput Run(const cv::Mat& bgr, MattingTimingInfo* timing = nullptr) const override;

private:
    float distance_threshold_;
};

// ── Registry ────────────────────────────────────────────────────────────────

/// Named registry of matting providers for runtime selection.
class CHROMAPRINT3D_API MattingRegistry {
public:
    /// Register a provider under the given key (overwrites if key exists).
    void Register(std::string key, MattingProviderPtr provider);

    /// Get a raw pointer to a provider (nullptr if not found).
    IMattingProvider* Get(const std::string& key) const;

    /// Get a shared_ptr to a provider (nullptr if not found).
    /// Suitable for capturing in lambda closures that outlive the lookup scope.
    MattingProviderPtr GetShared(const std::string& key) const;

    /// Check if a provider with the given key exists.
    bool Has(const std::string& key) const;

    /// List all registered provider keys.
    std::vector<std::string> Available() const;

private:
    std::unordered_map<std::string, MattingProviderPtr> providers_;
};

// ── Model discovery ─────────────────────────────────────────────────────────

/// Scan a directory for model files that have a matching JSON sidecar config.
/// Returns pairs of (model_file_path, parsed_config) sorted by filename stem.
CHROMAPRINT3D_API std::vector<std::pair<std::string, MattingModelConfig>>
DiscoverMattingModels(const std::string& directory);

// ── DL provider factory (available when infer module is linked) ─────────────

#if defined(CHROMAPRINT3D_HAS_INFER) && CHROMAPRINT3D_HAS_INFER

namespace infer {
class ISession;
using SessionPtr = std::unique_ptr<ISession>;
} // namespace infer

/// Create a deep-learning matting provider that wraps an inference session.
/// The returned provider takes ownership of the session.
CHROMAPRINT3D_API MattingProviderPtr CreateDLMattingProvider(const std::string& name,
                                                             infer::SessionPtr session,
                                                             const MattingModelConfig& config);

#endif // CHROMAPRINT3D_HAS_INFER

} // namespace ChromaPrint3D
