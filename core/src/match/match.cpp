#include "match.h"
#include "common.h"
#include "vec3.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <sys/types.h>

namespace ChromaPrint3D {

const uint8_t* RecipeMap::RecipeAt(int r, int c) const {
    if (r < 0 || c < 0 || r >= height || c >= width || color_layers <= 0) { return nullptr; }
    const size_t idx = static_cast<size_t>(r) * static_cast<size_t>(width) + static_cast<size_t>(c);
    const size_t offset = idx * static_cast<size_t>(color_layers);
    if (offset + static_cast<size_t>(color_layers) > recipes.size()) { return nullptr; }
    return &recipes[offset];
}

const uint8_t* RecipeMap::MaskAt(int r, int c) const {
    if (r < 0 || c < 0 || r >= height || c >= width) { return nullptr; }
    const size_t idx = static_cast<size_t>(r) * static_cast<size_t>(width) + static_cast<size_t>(c);
    if (idx >= mask.size()) { return nullptr; }
    return &mask[idx];
}

const Lab RecipeMap::ColorAt(int r, int c) const {
    if (r < 0 || c < 0 || r >= height || c >= width) { return Lab(); }
    const size_t idx = static_cast<size_t>(r) * static_cast<size_t>(width) + static_cast<size_t>(c);
    if (idx >= mapped_color.size()) { return Lab(); }
    return mapped_color[idx];
}

cv::Mat RecipeMap::ToBgrImage(uint8_t background_b, uint8_t background_g,
                              uint8_t background_r) const {
    if (width <= 0 || height <= 0) { return cv::Mat(); }

    const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
    if (mapped_color.size() < pixel_count) {
        throw std::runtime_error("mapped_color size mismatch");
    }
    if (!mask.empty() && mask.size() < pixel_count) {
        throw std::runtime_error("mask size mismatch");
    }

    cv::Mat bgr(height, width, CV_8UC3, cv::Scalar(background_b, background_g, background_r));
    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            const size_t idx =
                static_cast<size_t>(r) * static_cast<size_t>(width) + static_cast<size_t>(c);
            if (!mask.empty() && mask[idx] == 0) { continue; }

            Rgb rgb    = mapped_color[idx].ToRgb();
            uint8_t r8 = 0, g8 = 0, b8 = 0;
            rgb.ToRgb255(r8, g8, b8);
            bgr.at<cv::Vec3b>(r, c) = cv::Vec3b(b8, g8, r8);
        }
    }
    return bgr;
}

RecipeMap RecipeMap::MatchFromImage(const ImgProcResult& img, const ColorDB& db,
                                    const MatchConfig& cfg) {
    if (img.lab.empty()) { throw std::runtime_error("Image Lab data is empty"); }
    if (img.lab.type() != CV_32FC3) { throw std::runtime_error("Image Lab data must be CV_32FC3"); }
    if (img.lab.rows != img.height || img.lab.cols != img.width) {
        throw std::runtime_error("Image Lab size does not match ImgProcResult size");
    }
    if (!img.mask.empty() && (img.mask.rows != img.height || img.mask.cols != img.width)) {
        throw std::runtime_error("Image mask size does not match ImgProcResult size");
    }

    RecipeMap result;
    result.name         = img.name;
    result.width        = img.width;
    result.height       = img.height;
    result.color_layers = db.max_color_layers;
    result.num_channels = db.NumChannels();
    result.layer_order  = db.layer_order;
    result.recipes.assign(static_cast<size_t>(img.width) * static_cast<size_t>(img.height) *
                              static_cast<size_t>(db.max_color_layers),
                          0);
    result.mask.assign(static_cast<size_t>(img.width) * static_cast<size_t>(img.height), 0);
    result.mapped_color.assign(static_cast<size_t>(img.width) * static_cast<size_t>(img.height),
                               Lab());

    const bool has_mask = !img.mask.empty();

    const bool use_lab = (cfg.color_space == ColorSpace::Lab);
    if (!use_lab) {
        if (img.rgb.empty()) { throw std::runtime_error("Image RGB data is empty"); }
        if (img.rgb.type() != CV_32FC3) {
            throw std::runtime_error("Image RGB data must be CV_32FC3");
        }
        if (img.rgb.rows != img.height || img.rgb.cols != img.width) {
            throw std::runtime_error("Image RGB size does not match ImgProcResult size");
        }
    }

    const cv::Mat& target          = use_lab ? img.lab : img.rgb;
    const std::size_t color_layers = static_cast<std::size_t>(result.color_layers);
    const bool use_top_k           = cfg.k_candidates > 1;

    for (int r = 0; r < img.height; ++r) {
        const cv::Vec3f* target_row = target.ptr<cv::Vec3f>(r);
        const uint8_t* mask_row     = has_mask ? img.mask.ptr<uint8_t>(r) : nullptr;

        for (int c = 0; c < img.width; ++c) {
            const size_t idx =
                static_cast<size_t>(r) * static_cast<size_t>(img.width) + static_cast<size_t>(c);

            const uint8_t mask_value = has_mask ? mask_row[c] : static_cast<uint8_t>(255);
            result.mask[idx]         = (mask_value == 0) ? 0 : mask_value;
            if (mask_value == 0) { continue; }

            const cv::Vec3f& v = target_row[c];
            const Entry* entry = nullptr;
            if (use_lab) {
                Lab target_lab(v[0], v[1], v[2]);
                if (use_top_k) {
                    auto candidates = db.NearestEntries(target_lab, cfg.k_candidates);
                    if (candidates.empty()) { continue; }
                    entry = candidates.front();
                } else {
                    entry = &db.NearestEntry(target_lab);
                }
            } else {
                Rgb target_rgb(v[0], v[1], v[2]);
                if (use_top_k) {
                    auto candidates = db.NearestEntries(target_rgb, cfg.k_candidates);
                    if (candidates.empty()) { continue; }
                    entry = candidates.front();
                } else {
                    entry = &db.NearestEntry(target_rgb);
                }
            }

            result.mapped_color[idx] = entry->lab;

            if (color_layers == 0 || entry->recipe.empty()) { continue; }
            size_t offset = idx * color_layers;
            if (offset >= result.recipes.size()) { continue; }
            size_t copy_count = std::min(entry->recipe.size(), color_layers);
            if (offset + copy_count > result.recipes.size()) {
                copy_count = result.recipes.size() - offset;
            }
            std::copy_n(entry->recipe.begin(), copy_count, result.recipes.begin() + offset);
        }
    }

    return result;
}

} // namespace ChromaPrint3D