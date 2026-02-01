#pragma once

#include "common.h"
#include "vec3.h"
#include "imgproc.h"
#include "colorDB.h"

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <vector>

namespace ChromaPrint3D {

struct MatchConfig {
    int k_candidates = 1; // k <= 1 使用最邻近
    // TODO: Top-k match

    ColorSpace color_space = ColorSpace::Lab;
};

struct RecipeMap {
    int width        = 0;
    int height       = 0;
    int color_layers = 0;
    int num_channels = 0;

    LayerOrder layer_order = LayerOrder::Top2Bottom;

    std::vector<uint8_t> recipes;  // H * W * N
    std::vector<uint8_t> mask;     // H * W
    std::vector<Lab> mapped_color; // H * W

    const uint8_t* RecipeAt(int r, int c) const;
    const uint8_t* MaskAt(int r, int c) const;
    const Lab ColorAt(int r, int c) const;

    cv::Mat ToBgrImage(uint8_t background_b = 0, uint8_t background_g = 0,
                       uint8_t background_r = 0) const;

    static RecipeMap MatchFromImage(const ImgProcResult& img, const ColorDB& db,
                                    const MatchConfig& cfg = MatchConfig{});
};

} // namespace ChromaPrint3D