/// \file test_color_types.cpp
/// \brief Unit tests for the new first-class color types — `Rgb`, `Lab`,
///        `SrgbU8`, `Hsv`. Covers construction, accessors, arithmetic,
///        boundary helpers, hash specialisations, and Hsv equivalence
///        between FromRgb (linear) and FromSrgbU8 (gamma) entries.

#include <gtest/gtest.h>

#include "chromaprint3d/color.h"

#include <cmath>
#include <unordered_map>
#include <unordered_set>

using namespace ChromaPrint3D;

// ── Construction & accessors ────────────────────────────────────────────────

TEST(ColorTypes, RgbDefaultZero) {
    Rgb r;
    EXPECT_FLOAT_EQ(r.r(), 0.0f);
    EXPECT_FLOAT_EQ(r.g(), 0.0f);
    EXPECT_FLOAT_EQ(r.b(), 0.0f);
}

TEST(ColorTypes, RgbFieldAccess) {
    Rgb r(0.1f, 0.5f, 0.9f);
    EXPECT_FLOAT_EQ(r.r(), 0.1f);
    EXPECT_FLOAT_EQ(r.g(), 0.5f);
    EXPECT_FLOAT_EQ(r.b(), 0.9f);

    r.r() = 0.3f;
    EXPECT_FLOAT_EQ(r.r(), 0.3f);
}

TEST(ColorTypes, LabFieldAccess) {
    Lab lab(60.0f, 12.0f, -8.0f);
    EXPECT_FLOAT_EQ(lab.l(), 60.0f);
    EXPECT_FLOAT_EQ(lab.a(), 12.0f);
    EXPECT_FLOAT_EQ(lab.b(), -8.0f);
}

TEST(ColorTypes, SrgbU8DefaultZero) {
    SrgbU8 c;
    EXPECT_EQ(c.r, 0);
    EXPECT_EQ(c.g, 0);
    EXPECT_EQ(c.b, 0);
}

TEST(ColorTypes, SrgbU8FieldAccess) {
    SrgbU8 c{255, 128, 64};
    EXPECT_EQ(c.r, 255);
    EXPECT_EQ(c.g, 128);
    EXPECT_EQ(c.b, 64);
}

// ── Boundary helpers ────────────────────────────────────────────────────────

TEST(ColorTypes, RgbAsVec3fAndBack) {
    Rgb r(0.25f, 0.5f, 0.75f);
    Vec3f v   = r.AsVec3f();
    EXPECT_FLOAT_EQ(v.x, 0.25f);
    EXPECT_FLOAT_EQ(v.y, 0.5f);
    EXPECT_FLOAT_EQ(v.z, 0.75f);

    Rgb back = Rgb::FromVec3f(v);
    EXPECT_EQ(back, r);
}

TEST(ColorTypes, LabAsVec3fAndBack) {
    Lab lab(40.0f, 5.0f, -10.0f);
    Vec3f v = lab.AsVec3f();
    EXPECT_FLOAT_EQ(v.x, 40.0f);
    EXPECT_FLOAT_EQ(v.y, 5.0f);
    EXPECT_FLOAT_EQ(v.z, -10.0f);

    Lab back = Lab::FromVec3f(v);
    EXPECT_EQ(back, lab);
}

// ── Arithmetic & equality ───────────────────────────────────────────────────

TEST(ColorTypes, RgbEqualityIsValueBased) {
    Rgb a(0.1f, 0.2f, 0.3f);
    Rgb b(0.1f, 0.2f, 0.3f);
    Rgb c(0.1f, 0.2f, 0.4f);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(ColorTypes, LabEqualityIsValueBased) {
    Lab a(10.0f, 20.0f, 30.0f);
    Lab b(10.0f, 20.0f, 30.0f);
    Lab c(10.0f, 20.0f, 30.5f);
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(ColorTypes, SrgbU8Equality) {
    SrgbU8 a{1, 2, 3};
    SrgbU8 b{1, 2, 3};
    SrgbU8 c{1, 2, 4};
    EXPECT_EQ(a, b);
    EXPECT_NE(a, c);
}

TEST(ColorTypes, RgbAdditionAndScaling) {
    Rgb a(0.1f, 0.2f, 0.3f);
    Rgb b(0.05f, 0.05f, 0.05f);
    Rgb sum = a + b;
    EXPECT_FLOAT_EQ(sum.r(), 0.15f);

    Rgb scaled = a * 2.0f;
    EXPECT_FLOAT_EQ(scaled.r(), 0.2f);

    Rgb prefix = 2.0f * a;
    EXPECT_FLOAT_EQ(prefix.b(), 0.6f);
}

TEST(ColorTypes, RgbClamp01) {
    Rgb out_of_range(-0.1f, 1.5f, 0.5f);
    Rgb clamped = Rgb::Clamp01(out_of_range);
    EXPECT_FLOAT_EQ(clamped.r(), 0.0f);
    EXPECT_FLOAT_EQ(clamped.g(), 1.0f);
    EXPECT_FLOAT_EQ(clamped.b(), 0.5f);
}

TEST(ColorTypes, RgbLerpEndpoints) {
    Rgb a(0.0f, 0.0f, 0.0f);
    Rgb b(1.0f, 1.0f, 1.0f);
    EXPECT_EQ(Rgb::Lerp(a, b, 0.0f), a);
    EXPECT_EQ(Rgb::Lerp(a, b, 1.0f), b);
    Rgb mid = Rgb::Lerp(a, b, 0.5f);
    EXPECT_FLOAT_EQ(mid.r(), 0.5f);
}

// ── std::hash specialisations ───────────────────────────────────────────────

TEST(ColorTypes, RgbStdHashEqualForEqualValues) {
    Rgb a(0.5f, 0.5f, 0.5f);
    Rgb b(0.5f, 0.5f, 0.5f);
    EXPECT_EQ(std::hash<Rgb>{}(a), std::hash<Rgb>{}(b));
}

TEST(ColorTypes, RgbUnorderedSet) {
    std::unordered_set<Rgb> set;
    set.insert(Rgb(0.1f, 0.2f, 0.3f));
    set.insert(Rgb(0.4f, 0.5f, 0.6f));
    set.insert(Rgb(0.1f, 0.2f, 0.3f)); // duplicate
    EXPECT_EQ(set.size(), 2u);
}

TEST(ColorTypes, SrgbU8StdHash) {
    std::unordered_set<SrgbU8> set;
    set.insert(SrgbU8{1, 2, 3});
    set.insert(SrgbU8{4, 5, 6});
    set.insert(SrgbU8{1, 2, 3}); // duplicate
    EXPECT_EQ(set.size(), 2u);
}

// ── Hsv: FromRgb (linear) vs FromSrgbU8 (gamma) ─────────────────────────────

TEST(ColorTypes, HsvFromRgbLinearCorrectness) {
    // Pure red in linear space should map to hue 0, sat 1, val 1.
    Hsv h = Hsv::FromRgb(Rgb(1.0f, 0.0f, 0.0f));
    EXPECT_NEAR(h.h(), 0.0f, 1e-3f);
    EXPECT_NEAR(h.s(), 1.0f, 1e-3f);
    EXPECT_NEAR(h.v(), 1.0f, 1e-3f);
}

TEST(ColorTypes, HsvFromSrgbU8MaxGreenIsHue120) {
    Hsv h = Hsv::FromSrgbU8(SrgbU8{0, 255, 0});
    EXPECT_NEAR(h.h(), 120.0f, 0.5f);
    EXPECT_NEAR(h.s(), 1.0f, 1e-3f);
    EXPECT_NEAR(h.v(), 1.0f, 1e-3f);
}

TEST(ColorTypes, HsvLinearAndSrgbU8DifferOnMidGrey) {
    // Mid-grey in linear is (0.5, 0.5, 0.5), which has v == 0.5.
    // Same byte value (128, 128, 128) is gamma-encoded; v should be ~0.502.
    // The two paths use the same algebraic formula but on different inputs,
    // so values should differ for non-extremes.
    Hsv lin = Hsv::FromRgb(Rgb(0.5f, 0.5f, 0.5f));
    EXPECT_NEAR(lin.v(), 0.5f, 1e-3f);
    EXPECT_NEAR(lin.s(), 0.0f, 1e-3f);

    Hsv enc = Hsv::FromSrgbU8(SrgbU8{128, 128, 128});
    EXPECT_NEAR(enc.v(), 128.0f / 255.0f, 1e-3f);
    EXPECT_NEAR(enc.s(), 0.0f, 1e-3f);
}

TEST(ColorTypes, HsvRoundTripLinear) {
    Hsv h = Hsv::FromRgb(Rgb(0.3f, 0.6f, 0.2f));
    Rgb rgb = h.ToRgb();
    EXPECT_NEAR(rgb.r(), 0.3f, 1e-4f);
    EXPECT_NEAR(rgb.g(), 0.6f, 1e-4f);
    EXPECT_NEAR(rgb.b(), 0.2f, 1e-4f);
}
