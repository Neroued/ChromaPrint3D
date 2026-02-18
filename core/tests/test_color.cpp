#include <gtest/gtest.h>
#include "chromaprint3d/color.h"

#include <cmath>

using namespace ChromaPrint3D;

TEST(Color, SrgbGammaRoundTrip) {
    for (float v : {0.0f, 0.01f, 0.04045f, 0.1f, 0.5f, 0.9f, 1.0f}) {
        float linear = SrgbToLinear(v);
        float back   = LinearToSrgb(linear);
        EXPECT_NEAR(back, v, 1e-5f) << "v=" << v;
    }
}

TEST(Color, RgbLabRoundTrip) {
    Rgb original(0.2f, 0.5f, 0.8f);
    Lab lab = original.ToLab();
    Rgb back = lab.ToRgb();
    EXPECT_NEAR(back.r(), original.r(), 1e-4f);
    EXPECT_NEAR(back.g(), original.g(), 1e-4f);
    EXPECT_NEAR(back.b(), original.b(), 1e-4f);
}

TEST(Color, LabFromRgbConsistency) {
    Rgb rgb(0.3f, 0.6f, 0.1f);
    Lab lab1 = rgb.ToLab();
    Lab lab2 = Lab::FromRgb(rgb);
    EXPECT_FLOAT_EQ(lab1.l(), lab2.l());
    EXPECT_FLOAT_EQ(lab1.a(), lab2.a());
    EXPECT_FLOAT_EQ(lab1.b(), lab2.b());
}

TEST(Color, RgbFromLabConsistency) {
    Lab lab(50.0f, 20.0f, -30.0f);
    Rgb rgb1 = lab.ToRgb();
    Rgb rgb2 = Rgb::FromLab(lab);
    EXPECT_FLOAT_EQ(rgb1.r(), rgb2.r());
    EXPECT_FLOAT_EQ(rgb1.g(), rgb2.g());
    EXPECT_FLOAT_EQ(rgb1.b(), rgb2.b());
}

TEST(Color, DeltaE76SameColor) {
    Lab c(50.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(Lab::DeltaE76(c, c), 0.0f);
}

TEST(Color, DeltaE76KnownValue) {
    Lab a(50.0f, 0.0f, 0.0f);
    Lab b(50.0f, 3.0f, 4.0f);
    float de = Lab::DeltaE76(a, b);
    EXPECT_NEAR(de, 5.0f, 1e-5f);
}

TEST(Color, Rgb255RoundTrip) {
    uint8_t r = 128, g = 64, b = 200;
    Rgb rgb = Rgb::FromRgb255(r, g, b);
    uint8_t r2, g2, b2;
    rgb.ToRgb255(r2, g2, b2);
    EXPECT_EQ(r2, r);
    EXPECT_EQ(g2, g);
    EXPECT_EQ(b2, b);
}

TEST(Color, RgbArithmetic) {
    Rgb a(0.1f, 0.2f, 0.3f);
    Rgb b(0.4f, 0.5f, 0.6f);
    Rgb c = a + b;
    EXPECT_NEAR(c.r(), 0.5f, 1e-6f);
    EXPECT_NEAR(c.g(), 0.7f, 1e-6f);

    Rgb d = a * 2.0f;
    EXPECT_NEAR(d.r(), 0.2f, 1e-6f);
}

TEST(Color, LabArithmetic) {
    Lab a(50.0f, 10.0f, -20.0f);
    Lab b(30.0f, -5.0f, 15.0f);
    Lab c = a + b;
    EXPECT_NEAR(c.l(), 80.0f, 1e-6f);
    EXPECT_NEAR(c.a(), 5.0f, 1e-6f);
}

TEST(Color, LabLerp) {
    Lab a(0.0f, 0.0f, 0.0f);
    Lab b(100.0f, 50.0f, -50.0f);
    Lab mid = Lab::Lerp(a, b, 0.5f);
    EXPECT_NEAR(mid.l(), 50.0f, 1e-6f);
    EXPECT_NEAR(mid.a(), 25.0f, 1e-6f);
}

TEST(Color, BlackWhiteConversion) {
    Rgb black(0.0f, 0.0f, 0.0f);
    Lab black_lab = black.ToLab();
    EXPECT_NEAR(black_lab.l(), 0.0f, 0.1f);

    Rgb white(1.0f, 1.0f, 1.0f);
    Lab white_lab = white.ToLab();
    EXPECT_NEAR(white_lab.l(), 100.0f, 0.5f);
}
