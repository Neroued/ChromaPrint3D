#include <gtest/gtest.h>

#include "chromaprint3d/color_db.h"
#include "chromaprint3d/encoding.h"
#include "chromaprint3d/recipe_map.h"
#include "chromaprint3d/vector_preview.h"
#include "chromaprint3d/vector_proc.h"
#include "chromaprint3d/vector_recipe_map.h"

#include <opencv2/imgcodecs.hpp>

using namespace ChromaPrint3D;

TEST(LayerPreview, RasterLayerPreviewRendersExpectedChannelsAndPng) {
    RecipeMap map;
    map.width        = 2;
    map.height       = 1;
    map.color_layers = 2;
    map.num_channels = 2;
    map.recipes      = {0, 1, 1, 0};
    map.mask         = {255, 255};

    std::vector<Channel> palette = {
        Channel{"Red", "PLA"},
        Channel{"Blue", "PLA"},
    };

    cv::Mat layer0 = map.ToLayerBgrImage(0, palette, 255, 255, 255);
    ASSERT_FALSE(layer0.empty());
    ASSERT_EQ(layer0.rows, 1);
    ASSERT_EQ(layer0.cols, 2);
    EXPECT_EQ(layer0.at<cv::Vec3b>(0, 0), cv::Vec3b(0, 0, 255));
    EXPECT_EQ(layer0.at<cv::Vec3b>(0, 1), cv::Vec3b(255, 0, 0));

    cv::Mat layer1 = map.ToLayerBgrImage(1, palette, 255, 255, 255);
    ASSERT_FALSE(layer1.empty());
    EXPECT_EQ(layer1.at<cv::Vec3b>(0, 0), cv::Vec3b(255, 0, 0));
    EXPECT_EQ(layer1.at<cv::Vec3b>(0, 1), cv::Vec3b(0, 0, 255));

    std::vector<uint8_t> layer0_png = EncodePng(layer0);
    EXPECT_FALSE(layer0_png.empty());
}

TEST(LayerPreview, RasterPreviewAndLayerPreviewKeepTransparency) {
    RecipeMap map;
    map.width        = 2;
    map.height       = 1;
    map.color_layers = 1;
    map.num_channels = 2;
    map.recipes      = {0, 1};
    map.mask         = {255, 0};
    map.mapped_color = {
        Lab::FromRgb(SrgbU8(255, 0, 0).ToRgb()),
        Lab::FromRgb(SrgbU8(0, 255, 0).ToRgb()),
    };

    std::vector<Channel> palette = {
        Channel{"Red", "PLA"},
        Channel{"Blue", "PLA"},
    };

    cv::Mat preview_bgra = map.ToBgraImage();
    ASSERT_FALSE(preview_bgra.empty());
    ASSERT_EQ(preview_bgra.type(), CV_8UC4);
    EXPECT_EQ(preview_bgra.at<cv::Vec4b>(0, 0)[3], 255);
    EXPECT_EQ(preview_bgra.at<cv::Vec4b>(0, 1)[3], 0);

    std::vector<uint8_t> preview_png = EncodePng(preview_bgra);
    ASSERT_FALSE(preview_png.empty());
    cv::Mat decoded_preview = cv::imdecode(preview_png, cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(decoded_preview.empty());
    ASSERT_EQ(decoded_preview.type(), CV_8UC4);
    EXPECT_EQ(decoded_preview.at<cv::Vec4b>(0, 1)[3], 0);

    cv::Mat layer_bgra = map.ToLayerBgraImage(0, palette);
    ASSERT_FALSE(layer_bgra.empty());
    ASSERT_EQ(layer_bgra.type(), CV_8UC4);
    EXPECT_EQ(layer_bgra.at<cv::Vec4b>(0, 0), cv::Vec4b(0, 0, 255, 255));
    EXPECT_EQ(layer_bgra.at<cv::Vec4b>(0, 1)[3], 0);

    std::vector<uint8_t> layer_png = EncodePng(layer_bgra);
    ASSERT_FALSE(layer_png.empty());
    cv::Mat decoded_layer = cv::imdecode(layer_png, cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(decoded_layer.empty());
    ASSERT_EQ(decoded_layer.type(), CV_8UC4);
    EXPECT_EQ(decoded_layer.at<cv::Vec4b>(0, 1)[3], 0);
}

TEST(LayerPreview, VectorLayerPreviewPngsContainAllLayers) {
    VectorProcResult vector_result;
    vector_result.width_mm  = 10.0f;
    vector_result.height_mm = 10.0f;

    VectorShape shape;
    shape.fill_type = FillType::Solid;
    shape.contours.push_back(
        {Vec2f(0.0f, 0.0f), Vec2f(10.0f, 0.0f), Vec2f(10.0f, 10.0f), Vec2f(0.0f, 10.0f)});
    vector_result.shapes.push_back(shape);

    VectorRecipeMap recipe_map;
    recipe_map.color_layers = 2;
    recipe_map.num_channels = 2;
    recipe_map.entries.push_back(VectorRecipeMap::ShapeEntry{
        0, std::vector<uint8_t>{0, 1}, Lab::FromRgb(SrgbU8(255, 255, 255).ToRgb()), false});

    std::vector<Channel> palette = {
        Channel{"Green", "PLA"},
        Channel{"Magenta", "PLA"},
    };

    std::vector<std::vector<uint8_t>> pngs =
        RenderVectorLayerPreviewPngs(vector_result, recipe_map, palette, 1.0f);
    ASSERT_EQ(pngs.size(), 2u);
    EXPECT_FALSE(pngs[0].empty());
    EXPECT_FALSE(pngs[1].empty());

    cv::Mat layer0 = cv::imdecode(pngs[0], cv::IMREAD_COLOR);
    cv::Mat layer1 = cv::imdecode(pngs[1], cv::IMREAD_COLOR);
    ASSERT_FALSE(layer0.empty());
    ASSERT_FALSE(layer1.empty());
    EXPECT_EQ(layer0.rows, 10);
    EXPECT_EQ(layer0.cols, 10);
    EXPECT_EQ(layer1.rows, 10);
    EXPECT_EQ(layer1.cols, 10);

    EXPECT_EQ(layer0.at<cv::Vec3b>(5, 5), cv::Vec3b(0, 255, 0));
    EXPECT_EQ(layer1.at<cv::Vec3b>(5, 5), cv::Vec3b(255, 0, 255));
}

TEST(LayerPreview, VectorPreviewAndLayerPreviewKeepTransparentBackground) {
    VectorProcResult vector_result;
    vector_result.width_mm  = 10.0f;
    vector_result.height_mm = 10.0f;

    VectorShape shape;
    shape.fill_type = FillType::Solid;
    shape.contours.push_back(
        {Vec2f(0.0f, 0.0f), Vec2f(5.0f, 0.0f), Vec2f(5.0f, 5.0f), Vec2f(0.0f, 5.0f)});
    vector_result.shapes.push_back(shape);

    VectorRecipeMap recipe_map;
    recipe_map.color_layers = 1;
    recipe_map.num_channels = 1;
    recipe_map.entries.push_back(VectorRecipeMap::ShapeEntry{
        0, std::vector<uint8_t>{0}, Lab::FromRgb(SrgbU8(255, 0, 0).ToRgb()), false});

    std::vector<Channel> palette = {
        Channel{"Red", "PLA"},
    };

    std::vector<uint8_t> preview_png =
        RenderVectorPreviewPng(vector_result, recipe_map, palette, 1.0f);
    ASSERT_FALSE(preview_png.empty());
    cv::Mat preview = cv::imdecode(preview_png, cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(preview.empty());
    ASSERT_EQ(preview.type(), CV_8UC4);
    EXPECT_EQ(preview.at<cv::Vec4b>(2, 2)[3], 255);
    EXPECT_EQ(preview.at<cv::Vec4b>(8, 8)[3], 0);

    std::vector<uint8_t> layer_png =
        RenderVectorLayerPreviewPng(vector_result, recipe_map, palette, 0, 1.0f);
    ASSERT_FALSE(layer_png.empty());
    cv::Mat layer = cv::imdecode(layer_png, cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(layer.empty());
    ASSERT_EQ(layer.type(), CV_8UC4);
    EXPECT_EQ(layer.at<cv::Vec4b>(2, 2), cv::Vec4b(0, 0, 255, 255));
    EXPECT_EQ(layer.at<cv::Vec4b>(8, 8)[3], 0);
}
