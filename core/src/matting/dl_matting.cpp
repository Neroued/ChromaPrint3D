#include "chromaprint3d/matting.h"

#include <chromaprint3d/infer/infer.h>
#include <spdlog/spdlog.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <chrono>

namespace ChromaPrint3D {

namespace {

infer::ChannelOrder ParseChannelOrder(const std::string& s) {
    if (s == "bgr" || s == "BGR") return infer::ChannelOrder::kBGR;
    return infer::ChannelOrder::kRGB;
}

infer::LayoutOrder ParseLayout(const std::string& s) {
    if (s == "nhwc" || s == "NHWC") return infer::LayoutOrder::kNHWC;
    return infer::LayoutOrder::kNCHW;
}

infer::PreprocessConfig BuildPreprocessConfig(const MattingModelConfig& cfg) {
    infer::PreprocessConfig prep;
    prep.target_width      = cfg.input_width;
    prep.target_height     = cfg.input_height;
    prep.keep_aspect_ratio = true;
    prep.pad_value         = 0;
    prep.channel_order     = ParseChannelOrder(cfg.channel_order);
    prep.layout            = ParseLayout(cfg.layout);
    prep.normalize         = true;
    prep.scale             = cfg.normalize_scale;
    prep.mean              = cfg.normalize_mean;
    prep.std               = cfg.normalize_std;
    return prep;
}

} // namespace

class DLMattingProvider : public IMattingProvider {
public:
    DLMattingProvider(std::string name, std::shared_ptr<infer::ISession> session,
                      MattingModelConfig config)
        : name_(std::move(name)), session_(std::move(session)), config_(std::move(config)),
          prep_(BuildPreprocessConfig(config_)) {}

    std::string Name() const override { return name_; }

    std::string Description() const override { return config_.description; }

    MattingOutput Run(const cv::Mat& bgr, MattingTimingInfo* timing = nullptr) const override {
        if (bgr.empty()) return {};

        using Clock = std::chrono::steady_clock;
        auto t0     = Clock::now();

        infer::PreprocessMeta meta;
        infer::Tensor input = infer::PreprocessImage(bgr, prep_, &meta);
        auto t1             = Clock::now();

        auto outputs = session_->Run({input});
        auto t2      = Clock::now();

        if (config_.output_index >= static_cast<int>(outputs.size())) {
            spdlog::error("DLMattingProvider '{}': output_index {} out of range (got {} outputs)",
                          name_, config_.output_index, outputs.size());
            cv::Mat fallback(bgr.rows, bgr.cols, CV_8UC1, cv::Scalar(255));
            return {fallback, {}};
        }

        const auto& out_tensor = outputs[config_.output_index];
        cv::Mat alpha          = infer::PostprocessAlpha(out_tensor, meta);
        cv::Mat mask;
        cv::threshold(alpha, mask, static_cast<double>(config_.threshold) * 255.0, 255.0,
                      cv::THRESH_BINARY);
        auto t3 = Clock::now();

        auto ms       = [](auto d) { return std::chrono::duration<double, std::milli>(d).count(); };
        double pre_ms = ms(t1 - t0);
        double inf_ms = ms(t2 - t1);
        double post_ms = ms(t3 - t2);
        double tot_ms  = ms(t3 - t0);

        spdlog::debug("DLMatting '{}': preprocess={:.1f}ms, inference={:.1f}ms, "
                      "postprocess={:.1f}ms, total={:.1f}ms",
                      name_, pre_ms, inf_ms, post_ms, tot_ms);

        if (timing) {
            timing->preprocess_ms  = pre_ms;
            timing->inference_ms   = inf_ms;
            timing->postprocess_ms = post_ms;
            timing->total_ms       = tot_ms;
        }

        return {mask, alpha};
    }

private:
    std::string name_;
    std::shared_ptr<infer::ISession> session_;
    MattingModelConfig config_;
    infer::PreprocessConfig prep_;
};

MattingProviderPtr CreateDLMattingProvider(const std::string& name, infer::SessionPtr session,
                                           const MattingModelConfig& config) {
    auto device = session->GetDevice();
    spdlog::info("Creating DL matting provider '{}' on {}", name, device.ToString());
    return std::make_shared<DLMattingProvider>(
        name, std::shared_ptr<infer::ISession>(std::move(session)), config);
}

} // namespace ChromaPrint3D
