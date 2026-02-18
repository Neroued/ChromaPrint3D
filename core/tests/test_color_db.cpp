#include <gtest/gtest.h>
#include "chromaprint3d/color_db.h"

#include <cstddef>

using namespace ChromaPrint3D;

static ColorDB MakeSimpleDB() {
    ColorDB db;
    db.name             = "test_db";
    db.max_color_layers = 5;
    db.base_layers      = 3;
    db.base_channel_idx = 0;
    db.layer_height_mm  = 0.08f;
    db.line_width_mm    = 0.42f;
    db.layer_order      = LayerOrder::Top2Bottom;

    Channel white;
    white.color    = "White";
    white.material = "PLA";
    Channel red;
    red.color    = "Red";
    red.material = "PLA";
    db.palette = {white, red};

    Entry e1;
    e1.lab    = Lab(50.0f, 20.0f, -10.0f);
    e1.recipe = {0, 0, 1, 1, 0};
    db.entries.push_back(e1);

    Entry e2;
    e2.lab    = Lab(70.0f, -5.0f, 30.0f);
    e2.recipe = {1, 1, 0, 0, 1};
    db.entries.push_back(e2);

    return db;
}

TEST(ColorDB, BasicProperties) {
    ColorDB db = MakeSimpleDB();
    EXPECT_EQ(db.name, "test_db");
    EXPECT_EQ(db.NumChannels(), 2u);
    EXPECT_EQ(db.entries.size(), 2u);
}

TEST(ColorDB, NearestEntry) {
    ColorDB db = MakeSimpleDB();

    const Entry& nearest = db.NearestEntry(Lab(51.0f, 21.0f, -11.0f));
    EXPECT_NEAR(nearest.lab.l(), 50.0f, 0.1f);
}

TEST(ColorDB, NearestEntries) {
    ColorDB db = MakeSimpleDB();

    auto results = db.NearestEntries(Lab(60.0f, 0.0f, 0.0f), 2);
    EXPECT_EQ(results.size(), 2u);
    EXPECT_NE(results[0], nullptr);
    EXPECT_NE(results[1], nullptr);
}

TEST(ColorDB, EmptyDB) {
    ColorDB db;
    db.name             = "empty";
    db.max_color_layers = 5;
    db.palette.push_back(Channel{"White", "PLA"});
    EXPECT_TRUE(db.entries.empty());
    EXPECT_EQ(db.NumChannels(), 1u);
}
