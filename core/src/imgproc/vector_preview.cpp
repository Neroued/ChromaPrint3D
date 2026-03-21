#include "chromaprint3d/vector_preview.h"
#include "chromaprint3d/color.h"
#include "chromaprint3d/encoding.h"
#include "detail/layer_preview_color.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace ChromaPrint3D {

namespace {

void FillShapeOnImage(cv::Mat& img, const VectorShape& shape, const cv::Scalar& color, float ox,
                      float oy, float scale) {
    std::vector<std::vector<cv::Point>> all_pts;
    all_pts.reserve(shape.contours.size());

    for (const Contour& contour : shape.contours) {
        if (contour.size() < 3) { continue; }

        std::vector<cv::Point> pts;
        pts.reserve(contour.size());
        for (const Vec2f& p : contour) {
            pts.emplace_back(static_cast<int>(std::round((p.x - ox) * scale)),
                             static_cast<int>(std::round((p.y - oy) * scale)));
        }
        all_pts.push_back(std::move(pts));
    }

    if (!all_pts.empty()) { cv::fillPoly(img, all_pts, color); }
}

} // namespace

cv::Mat RenderVectorPreview(const VectorProcResult& result, const VectorRecipeMap& recipe_map,
                            const std::vector<Channel>& /*palette*/, float pixels_per_mm) {
    if (result.shapes.empty() || pixels_per_mm <= 0.0f) { return {}; }

    int w = std::max(1, static_cast<int>(std::ceil(result.width_mm * pixels_per_mm)));
    int h = std::max(1, static_cast<int>(std::ceil(result.height_mm * pixels_per_mm)));

    cv::Mat img(h, w, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    std::unordered_map<int, const VectorRecipeMap::ShapeEntry*> entry_map;
    for (const auto& entry : recipe_map.entries) { entry_map[entry.shape_idx] = &entry; }

    for (size_t i = 0; i < result.shapes.size(); ++i) {
        const VectorShape& shape = result.shapes[i];

        cv::Scalar color(200, 200, 200, 255);

        auto it = entry_map.find(static_cast<int>(i));
        if (it != entry_map.end()) {
            Lab lab = it->second->matched_color;
            Rgb rgb = lab.ToRgb();
            int br  = static_cast<int>(std::clamp(LinearToSrgb(rgb.r()) * 255.0f, 0.0f, 255.0f));
            int bg  = static_cast<int>(std::clamp(LinearToSrgb(rgb.g()) * 255.0f, 0.0f, 255.0f));
            int bb  = static_cast<int>(std::clamp(LinearToSrgb(rgb.b()) * 255.0f, 0.0f, 255.0f));
            color   = cv::Scalar(bb, bg, br, 255);
        }

        FillShapeOnImage(img, shape, color, 0.0f, 0.0f, pixels_per_mm);
    }

    if (result.y_flipped) { cv::flip(img, img, 0); }

    return img;
}

std::vector<uint8_t> RenderVectorPreviewPng(const VectorProcResult& result,
                                            const VectorRecipeMap& recipe_map,
                                            const std::vector<Channel>& palette,
                                            float pixels_per_mm) {
    cv::Mat img = RenderVectorPreview(result, recipe_map, palette, pixels_per_mm);
    if (img.empty()) { return {}; }
    return EncodePng(img);
}

cv::Mat RenderVectorSourceMask(const VectorProcResult& result, const VectorRecipeMap& recipe_map,
                               float pixels_per_mm) {
    if (result.shapes.empty() || pixels_per_mm <= 0.0f) { return {}; }

    int w = std::max(1, static_cast<int>(std::ceil(result.width_mm * pixels_per_mm)));
    int h = std::max(1, static_cast<int>(std::ceil(result.height_mm * pixels_per_mm)));

    cv::Mat mask(h, w, CV_8UC1, cv::Scalar(0));

    std::unordered_map<int, const VectorRecipeMap::ShapeEntry*> entry_map;
    for (const auto& entry : recipe_map.entries) { entry_map[entry.shape_idx] = &entry; }

    for (size_t i = 0; i < result.shapes.size(); ++i) {
        const VectorShape& shape = result.shapes[i];
        auto it                  = entry_map.find(static_cast<int>(i));
        const bool from_model    = (it != entry_map.end()) && it->second->from_model;
        FillShapeOnImage(mask, shape, cv::Scalar(from_model ? 255 : 0), 0.0f, 0.0f, pixels_per_mm);
    }

    if (result.y_flipped) { cv::flip(mask, mask, 0); }

    return mask;
}

std::vector<uint8_t> RenderVectorSourceMaskPng(const VectorProcResult& result,
                                               const VectorRecipeMap& recipe_map,
                                               float pixels_per_mm) {
    cv::Mat mask = RenderVectorSourceMask(result, recipe_map, pixels_per_mm);
    if (mask.empty()) { return {}; }
    return EncodePng(mask);
}

cv::Mat RenderVectorLayerPreview(const VectorProcResult& result, const VectorRecipeMap& recipe_map,
                                 const std::vector<Channel>& palette, int layer_idx,
                                 float pixels_per_mm) {
    if (result.shapes.empty() || pixels_per_mm <= 0.0f) { return {}; }
    if (layer_idx < 0 || layer_idx >= recipe_map.color_layers) { return {}; }

    int w = std::max(1, static_cast<int>(std::ceil(result.width_mm * pixels_per_mm)));
    int h = std::max(1, static_cast<int>(std::ceil(result.height_mm * pixels_per_mm)));
    cv::Mat img(h, w, CV_8UC4, cv::Scalar(0, 0, 0, 0));

    std::unordered_map<int, const VectorRecipeMap::ShapeEntry*> entry_map;
    for (const auto& entry : recipe_map.entries) { entry_map[entry.shape_idx] = &entry; }

    for (size_t i = 0; i < result.shapes.size(); ++i) {
        const VectorShape& shape = result.shapes[i];
        cv::Scalar color(200, 200, 200, 255);

        auto it = entry_map.find(static_cast<int>(i));
        if (it != entry_map.end()) {
            const auto& recipe = it->second->recipe;
            if (static_cast<std::size_t>(layer_idx) < recipe.size()) {
                const std::size_t channel_idx = recipe[static_cast<std::size_t>(layer_idx)];
                if (channel_idx < palette.size()) {
                    const cv::Vec3b bgr =
                        detail::PaletteColorLiteralToBgr(palette[channel_idx].color);
                    color = cv::Scalar(bgr[0], bgr[1], bgr[2], 255);
                } else {
                    color = cv::Scalar(127, 127, 127, 255);
                }
            }
        }
        FillShapeOnImage(img, shape, color, 0.0f, 0.0f, pixels_per_mm);
    }

    if (result.y_flipped) { cv::flip(img, img, 0); }
    return img;
}

std::vector<uint8_t> RenderVectorLayerPreviewPng(const VectorProcResult& result,
                                                 const VectorRecipeMap& recipe_map,
                                                 const std::vector<Channel>& palette, int layer_idx,
                                                 float pixels_per_mm) {
    cv::Mat img = RenderVectorLayerPreview(result, recipe_map, palette, layer_idx, pixels_per_mm);
    if (img.empty()) return {};
    return EncodePng(img);
}

std::vector<std::vector<uint8_t>> RenderVectorLayerPreviewPngs(const VectorProcResult& result,
                                                               const VectorRecipeMap& recipe_map,
                                                               const std::vector<Channel>& palette,
                                                               float pixels_per_mm) {
    std::vector<std::vector<uint8_t>> out;
    if (recipe_map.color_layers <= 0) return out;
    out.reserve(static_cast<std::size_t>(recipe_map.color_layers));
    for (int layer_idx = 0; layer_idx < recipe_map.color_layers; ++layer_idx) {
        out.push_back(
            RenderVectorLayerPreviewPng(result, recipe_map, palette, layer_idx, pixels_per_mm));
    }
    return out;
}

cv::Mat RenderVectorRegionIds(const VectorProcResult& result, const VectorRecipeMap& recipe_map,
                              float pixels_per_mm) {
    if (result.shapes.empty() || pixels_per_mm <= 0.0f) { return {}; }

    int w = std::max(1, static_cast<int>(std::ceil(result.width_mm * pixels_per_mm)));
    int h = std::max(1, static_cast<int>(std::ceil(result.height_mm * pixels_per_mm)));

    constexpr int32_t kBackground = static_cast<int32_t>(0xFFFFFFFF);
    cv::Mat ids(h, w, CV_32SC1, cv::Scalar(kBackground));

    for (std::size_t ei = 0; ei < recipe_map.entries.size(); ++ei) {
        const auto& entry          = recipe_map.entries[ei];
        const VectorShape& shape   = result.shapes[static_cast<std::size_t>(entry.shape_idx)];
        const cv::Scalar entry_val = cv::Scalar(static_cast<int32_t>(ei));
        FillShapeOnImage(ids, shape, entry_val, 0.0f, 0.0f, pixels_per_mm);
    }

    if (result.y_flipped) { cv::flip(ids, ids, 0); }

    return ids;
}

} // namespace ChromaPrint3D
