#include <gtest/gtest.h>

#include "chromaprint3d/color.h"
#include "chromaprint3d/color_db.h"
#include "chromaprint3d/print_profile.h"
#include "chromaprint3d/vector_proc.h"
#include "chromaprint3d/vector_recipe_map.h"

#include <string>
#include <vector>

using namespace ChromaPrint3D;

namespace {

ColorDB MakePaletteReorderedDb() {
    ColorDB db;
    db.name             = "vector_match_test_db";
    db.max_color_layers = 5;
    db.base_layers      = 3;
    db.base_channel_idx = 0; // white in source DB.
    db.layer_height_mm  = 0.08f;
    db.line_width_mm    = 0.42f;
    db.layer_order      = LayerOrder::Top2Bottom;

    // Intentionally non-lexicographic order to trigger profile remap.
    db.palette = {Channel{"White", "PLA"}, Channel{"Black", "PLA"}};

    Entry e;
    e.lab    = Lab::FromRgb(Rgb::FromRgb255(255, 255, 255));
    e.recipe = {0, 0, 0, 0, 0}; // Source DB channel index 0 (White).
    db.entries.push_back(e);
    return db;
}

VectorProcResult MakeSingleColorVector(const Rgb& color) {
    VectorProcResult result;
    result.width_mm  = 10.0f;
    result.height_mm = 10.0f;

    VectorShape shape;
    shape.fill_type  = FillType::Solid;
    shape.fill_color = color;
    result.shapes.push_back(shape);
    return result;
}

int FindChannelIndexByName(const PrintProfile& profile, const std::string& color_name) {
    for (std::size_t i = 0; i < profile.palette.size(); ++i) {
        if (profile.palette[i].color == color_name) { return static_cast<int>(i); }
    }
    return -1;
}

ModelPackage MakeSingleCandidateModel(const PrintProfile& profile, uint8_t recipe_channel,
                                      const Lab& pred_lab, float default_threshold = 5.0f,
                                      float default_margin = 0.7f) {
    ModelPackage pack;
    pack.name              = "vector_match_test_model";
    pack.default_threshold = default_threshold;
    pack.default_margin    = default_margin;
    for (const auto& channel : profile.palette) {
        pack.channel_keys.push_back(channel.color + "|" + channel.material);
    }

    ModelLayerPackage lpkg;
    lpkg.layer_height_mm = profile.layer_height_mm;
    lpkg.color_layers    = profile.color_layers;
    lpkg.layer_order     = profile.layer_order;
    lpkg.candidate_recipes.assign(static_cast<std::size_t>(profile.color_layers), recipe_channel);
    lpkg.pred_lab.push_back(pred_lab);
    pack.layer_packages.push_back(std::move(lpkg));
    pack.BuildIndex();
    return pack;
}

} // namespace

TEST(VectorMatch, RemapsRecipeToProfilePaletteOrder) {
    ColorDB db = MakePaletteReorderedDb();
    std::vector<ColorDB> dbs{db};
    PrintProfile profile = PrintProfile::BuildFromColorDBs(dbs, 5, 0.08f);
    VectorProcResult vec = MakeSingleColorVector(Rgb::FromRgb255(255, 255, 255));

    MatchConfig cfg;
    cfg.color_space = ColorSpace::Lab;

    VectorRecipeMap map = VectorRecipeMap::Match(vec, dbs, profile, cfg);
    ASSERT_EQ(map.entries.size(), 1u);
    ASSERT_EQ(map.entries[0].recipe.size(), static_cast<std::size_t>(profile.color_layers));

    const int white_idx = FindChannelIndexByName(profile, "White");
    ASSERT_GE(white_idx, 0);

    for (uint8_t ch : map.entries[0].recipe) { EXPECT_EQ(ch, static_cast<uint8_t>(white_idx)); }
}

TEST(VectorMatch, ModelOnlyUsesModelRecipeForVectorSolidColor) {
    ColorDB db = MakePaletteReorderedDb();
    std::vector<ColorDB> dbs{db};
    PrintProfile profile = PrintProfile::BuildFromColorDBs(dbs, 5, 0.08f);
    VectorProcResult vec = MakeSingleColorVector(Rgb::FromRgb255(255, 255, 255));

    const int black_idx = FindChannelIndexByName(profile, "Black");
    ASSERT_GE(black_idx, 0);
    ModelPackage model = MakeSingleCandidateModel(profile, static_cast<uint8_t>(black_idx),
                                                  Lab::FromRgb(Rgb::FromRgb255(255, 255, 255)));

    MatchConfig cfg;
    cfg.color_space = ColorSpace::Lab;
    ModelGateConfig gate;
    gate.enable     = true;
    gate.model_only = true;

    MatchStats stats;
    VectorRecipeMap map = VectorRecipeMap::Match(vec, dbs, profile, cfg, &model, gate, &stats);
    ASSERT_EQ(map.entries.size(), 1u);
    for (uint8_t ch : map.entries[0].recipe) { EXPECT_EQ(ch, static_cast<uint8_t>(black_idx)); }
    EXPECT_EQ(stats.clusters_total, 1);
    EXPECT_EQ(stats.db_only, 0);
    EXPECT_EQ(stats.model_fallback, 1);
    EXPECT_EQ(stats.model_queries, 1);
}

TEST(VectorMatch, ThresholdFallbackUsesModelWhenEnabled) {
    ColorDB db = MakePaletteReorderedDb();
    std::vector<ColorDB> dbs{db};
    PrintProfile profile = PrintProfile::BuildFromColorDBs(dbs, 5, 0.08f);
    VectorProcResult vec = MakeSingleColorVector(Rgb::FromRgb255(128, 128, 128));

    const int black_idx = FindChannelIndexByName(profile, "Black");
    const int white_idx = FindChannelIndexByName(profile, "White");
    ASSERT_GE(black_idx, 0);
    ASSERT_GE(white_idx, 0);
    ModelPackage model = MakeSingleCandidateModel(profile, static_cast<uint8_t>(black_idx),
                                                  Lab::FromRgb(Rgb::FromRgb255(128, 128, 128)));

    MatchConfig cfg;
    cfg.color_space = ColorSpace::Lab;
    ModelGateConfig gate;
    gate.enable    = true;
    gate.threshold = 0.0f;
    gate.margin    = 0.0f;

    MatchStats stats;
    VectorRecipeMap map = VectorRecipeMap::Match(vec, dbs, profile, cfg, &model, gate, &stats);
    ASSERT_EQ(map.entries.size(), 1u);
    for (uint8_t ch : map.entries[0].recipe) { EXPECT_EQ(ch, static_cast<uint8_t>(black_idx)); }
    EXPECT_NE(black_idx, white_idx);
    EXPECT_EQ(stats.clusters_total, 1);
    EXPECT_EQ(stats.db_only, 0);
    EXPECT_EQ(stats.model_fallback, 1);
    EXPECT_EQ(stats.model_queries, 1);
}

TEST(VectorMatch, HighThresholdKeepsDbResultWithoutModelQuery) {
    ColorDB db = MakePaletteReorderedDb();
    std::vector<ColorDB> dbs{db};
    PrintProfile profile = PrintProfile::BuildFromColorDBs(dbs, 5, 0.08f);
    VectorProcResult vec = MakeSingleColorVector(Rgb::FromRgb255(128, 128, 128));

    const int black_idx = FindChannelIndexByName(profile, "Black");
    const int white_idx = FindChannelIndexByName(profile, "White");
    ASSERT_GE(black_idx, 0);
    ASSERT_GE(white_idx, 0);
    ModelPackage model = MakeSingleCandidateModel(profile, static_cast<uint8_t>(black_idx),
                                                  Lab::FromRgb(Rgb::FromRgb255(128, 128, 128)));

    MatchConfig cfg;
    cfg.color_space = ColorSpace::Lab;
    ModelGateConfig gate;
    gate.enable    = true;
    gate.threshold = 1000.0f;
    gate.margin    = 0.0f;

    MatchStats stats;
    VectorRecipeMap map = VectorRecipeMap::Match(vec, dbs, profile, cfg, &model, gate, &stats);
    ASSERT_EQ(map.entries.size(), 1u);
    for (uint8_t ch : map.entries[0].recipe) { EXPECT_EQ(ch, static_cast<uint8_t>(white_idx)); }
    EXPECT_EQ(stats.clusters_total, 1);
    EXPECT_EQ(stats.db_only, 1);
    EXPECT_EQ(stats.model_fallback, 0);
    EXPECT_EQ(stats.model_queries, 0);
}
