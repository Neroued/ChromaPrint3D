#include <gtest/gtest.h>

#include "chromaprint3d/color_db.h"
#include "chromaprint3d/print_profile.h"
#include "chromaprint3d/raster_proc.h"
#include "chromaprint3d/recipe_map.h"

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace ChromaPrint3D;

namespace {

ColorDB MakeTwoColorDb() {
    ColorDB db;
    db.name             = "raster_match_test_db";
    db.max_color_layers = 5;
    db.base_layers      = 3;
    db.base_channel_idx = 0;
    db.layer_height_mm  = 0.08f;
    db.line_width_mm    = 0.42f;
    db.layer_order      = LayerOrder::Top2Bottom;
    db.palette          = {Channel{"White", "PLA"}, Channel{"Black", "PLA"}};

    Entry white;
    white.lab    = Lab::FromRgb(Rgb::FromRgb255(255, 255, 255));
    white.recipe = {0, 0, 0, 0, 0};

    Entry black;
    black.lab    = Lab::FromRgb(Rgb::FromRgb255(0, 0, 0));
    black.recipe = {1, 1, 1, 1, 1};

    db.entries = {white, black};
    return db;
}

RasterProcResult MakeRaster(const std::vector<Rgb>& pixels, int width, int height,
                            const std::vector<uint8_t>& mask_values) {
    EXPECT_EQ(static_cast<int>(pixels.size()), width * height);
    EXPECT_EQ(static_cast<int>(mask_values.size()), width * height);

    RasterProcResult img;
    img.name   = "raster_match_case";
    img.width  = width;
    img.height = height;
    img.rgb    = cv::Mat(height, width, CV_32FC3);
    img.lab    = cv::Mat(height, width, CV_32FC3);
    img.mask   = cv::Mat(height, width, CV_8UC1);

    for (int r = 0; r < height; ++r) {
        cv::Vec3f* rgb_row = img.rgb.ptr<cv::Vec3f>(r);
        cv::Vec3f* lab_row = img.lab.ptr<cv::Vec3f>(r);
        uint8_t* mask_row  = img.mask.ptr<uint8_t>(r);
        for (int c = 0; c < width; ++c) {
            const std::size_t idx = static_cast<std::size_t>(r) * static_cast<std::size_t>(width) +
                                    static_cast<std::size_t>(c);
            const Rgb& rgb = pixels[idx];
            const Lab lab  = Lab::FromRgb(rgb);
            rgb_row[c]     = cv::Vec3f(rgb.r(), rgb.g(), rgb.b());
            lab_row[c]     = cv::Vec3f(lab.l(), lab.a(), lab.b());
            mask_row[c]    = mask_values[idx];
        }
    }
    return img;
}

int CountValidPixels(const cv::Mat& mask) {
    int valid = 0;
    for (int r = 0; r < mask.rows; ++r) {
        const uint8_t* row = mask.ptr<uint8_t>(r);
        for (int c = 0; c < mask.cols; ++c) {
            if (row[c] != 0) { ++valid; }
        }
    }
    return valid;
}

} // namespace

TEST(RasterMatch, SlicPathRespectsTransparentMaskPixels) {
    ColorDB db = MakeTwoColorDb();
    std::vector<ColorDB> dbs{db};
    PrintProfile profile = PrintProfile::BuildFromColorDBs(dbs, 5, 0.08f);

    const std::vector<Rgb> pixels = {
        Rgb::FromRgb255(255, 255, 255), Rgb::FromRgb255(0, 0, 0),
        Rgb::FromRgb255(255, 255, 255), Rgb::FromRgb255(0, 0, 0),
        Rgb::FromRgb255(255, 255, 255), Rgb::FromRgb255(0, 0, 0),
    };
    const std::vector<uint8_t> mask = {255, 255, 255, 255, 0, 255};
    RasterProcResult img            = MakeRaster(pixels, 3, 2, mask);

    MatchConfig cfg;
    cfg.cluster_method          = ClusterMethod::Slic;
    cfg.slic_target_superpixels = 4;
    cfg.slic_compactness        = 10.0f;
    cfg.slic_iterations         = 8;
    cfg.slic_min_region_ratio   = 0.2f;

    MatchStats stats;
    RecipeMap map = RecipeMap::MatchFromRaster(img, dbs, profile, cfg, nullptr, {}, &stats);

    ASSERT_EQ(map.width, 3);
    ASSERT_EQ(map.height, 2);
    ASSERT_EQ(static_cast<int>(map.mask.size()), 6);
    ASSERT_EQ(map.mask[4], 0); // masked-out pixel should remain transparent.
    EXPECT_GT(stats.clusters_total, 0);
}

TEST(RasterMatch, SlicTargetSuperpixelsOneFallsBackToPerPixelMatching) {
    ColorDB db = MakeTwoColorDb();
    std::vector<ColorDB> dbs{db};
    PrintProfile profile = PrintProfile::BuildFromColorDBs(dbs, 5, 0.08f);

    const std::vector<Rgb> pixels = {
        Rgb::FromRgb255(255, 255, 255),
        Rgb::FromRgb255(0, 0, 0),
        Rgb::FromRgb255(255, 255, 255),
        Rgb::FromRgb255(0, 0, 0),
    };
    const std::vector<uint8_t> mask = {255, 255, 255, 255};
    RasterProcResult img            = MakeRaster(pixels, 2, 2, mask);

    MatchConfig cfg;
    cfg.cluster_method          = ClusterMethod::Slic;
    cfg.slic_target_superpixels = 1;

    MatchStats stats;
    RecipeMap::MatchFromRaster(img, dbs, profile, cfg, nullptr, {}, &stats);
    EXPECT_EQ(stats.clusters_total, CountValidPixels(img.mask));
}

TEST(RasterMatch, KMeansAndSlicBothProduceValidClusterStats) {
    ColorDB db = MakeTwoColorDb();
    std::vector<ColorDB> dbs{db};
    PrintProfile profile = PrintProfile::BuildFromColorDBs(dbs, 5, 0.08f);

    const std::vector<Rgb> pixels = {
        Rgb::FromRgb255(255, 255, 255), Rgb::FromRgb255(255, 255, 255),
        Rgb::FromRgb255(0, 0, 0),       Rgb::FromRgb255(0, 0, 0),
        Rgb::FromRgb255(255, 255, 255), Rgb::FromRgb255(255, 255, 255),
        Rgb::FromRgb255(0, 0, 0),       Rgb::FromRgb255(0, 0, 0),
    };
    const std::vector<uint8_t> mask(8, 255);
    RasterProcResult img = MakeRaster(pixels, 4, 2, mask);

    MatchConfig kmeans_cfg;
    kmeans_cfg.cluster_method = ClusterMethod::KMeans;
    kmeans_cfg.cluster_count  = 2;

    MatchConfig slic_cfg;
    slic_cfg.cluster_method          = ClusterMethod::Slic;
    slic_cfg.slic_target_superpixels = 2;
    slic_cfg.slic_compactness        = 8.0f;
    slic_cfg.slic_iterations         = 6;
    slic_cfg.slic_min_region_ratio   = 0.2f;

    MatchStats kmeans_stats;
    MatchStats slic_stats;
    RecipeMap::MatchFromRaster(img, dbs, profile, kmeans_cfg, nullptr, {}, &kmeans_stats);
    RecipeMap::MatchFromRaster(img, dbs, profile, slic_cfg, nullptr, {}, &slic_stats);

    const int valid_pixels = CountValidPixels(img.mask);
    EXPECT_EQ(kmeans_stats.clusters_total, 2);
    EXPECT_GE(slic_stats.clusters_total, 1);
    EXPECT_LE(slic_stats.clusters_total, valid_pixels);
}
