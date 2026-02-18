#include <gtest/gtest.h>
#include "chromaprint3d/print_profile.h"
#include "match/detail/recipe_convert.h"
#include "match/detail/match_utils.h"

#include <vector>

using namespace ChromaPrint3D;
using namespace ChromaPrint3D::detail;

static PrintProfile MakeTestProfile() {
    PrintProfile profile;
    profile.mode                   = PrintMode::Mode0p08x5;
    profile.max_color_thickness_mm = 0.4f;
    profile.layer_height_mm        = 0.08f;
    profile.color_layers           = 5;
    profile.line_width_mm          = 0.42f;
    profile.base_layers            = 3;
    profile.base_channel_idx       = 0;
    profile.layer_order            = LayerOrder::Top2Bottom;

    Channel white{"White", "PLA"};
    Channel red{"Red", "PLA"};
    profile.palette = {white, red};
    return profile;
}

static ColorDB MakeTestDB() {
    ColorDB db;
    db.name             = "test";
    db.max_color_layers = 5;
    db.base_layers      = 3;
    db.base_channel_idx = 0;
    db.layer_height_mm  = 0.08f;
    db.line_width_mm    = 0.42f;
    db.layer_order      = LayerOrder::Top2Bottom;

    db.palette = {Channel{"White", "PLA"}, Channel{"Red", "PLA"}};

    Entry e;
    e.lab    = Lab(50.0f, 10.0f, -5.0f);
    e.recipe = {0, 0, 1, 1, 0};
    db.entries.push_back(e);
    return db;
}

TEST(RecipeConvert, SameLayerHeightConversion) {
    PrintProfile profile = MakeTestProfile();
    ColorDB db           = MakeTestDB();

    PreparedDB pdb;
    pdb.db = &db;
    pdb.source_to_target_channel = {0, 1};

    std::vector<uint8_t> out_recipe;
    bool ok = ConvertRecipeToProfile(db.entries[0], pdb, profile, out_recipe);
    EXPECT_TRUE(ok);
    EXPECT_EQ(out_recipe.size(), 5u);
    EXPECT_EQ(out_recipe[0], 0);
    EXPECT_EQ(out_recipe[2], 1);
}

TEST(RecipeConvert, EmptyRecipeReturnsFlase) {
    PrintProfile profile = MakeTestProfile();

    Entry empty_entry;
    empty_entry.lab = Lab(50.0f, 0.0f, 0.0f);

    ColorDB db = MakeTestDB();
    PreparedDB pdb;
    pdb.db = &db;
    pdb.source_to_target_channel = {0, 1};

    std::vector<uint8_t> out;
    EXPECT_FALSE(ConvertRecipeToProfile(empty_entry, pdb, profile, out));
}

TEST(RecipeConvert, UnmappedChannelReturnsFlase) {
    PrintProfile profile = MakeTestProfile();
    ColorDB db           = MakeTestDB();

    PreparedDB pdb;
    pdb.db = &db;
    pdb.source_to_target_channel = {0, -1};

    std::vector<uint8_t> out;
    bool ok = ConvertRecipeToProfile(db.entries[0], pdb, profile, out);
    EXPECT_FALSE(ok);
}

TEST(RecipeConvert, PrepareDBsBasic) {
    PrintProfile profile = MakeTestProfile();
    ColorDB db           = MakeTestDB();

    std::vector<ColorDB> dbs = {db};
    auto prepared = PrepareDBs(dbs, profile);
    EXPECT_EQ(prepared.size(), 1u);
    EXPECT_EQ(prepared[0].source_to_target_channel.size(), 2u);
}
