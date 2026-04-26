/// \file test_color_parse.cpp
/// \brief Unit tests for `SrgbU8::FromHex` / `ToHex`, `NamedColorLookup`,
///        `HashFallbackColor`, `FallbackColorByIndex`, and the unified
///        `ResolveColorLiteral` entry.

#include <gtest/gtest.h>

#include "chromaprint3d/color.h"

#include <string>

using namespace ChromaPrint3D;

// ── SrgbU8::FromHex strict input contract ───────────────────────────────────

TEST(ColorParse, FromHexHashSixDigits) {
    auto c = SrgbU8::FromHex("#FF8000");
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->r, 0xFF);
    EXPECT_EQ(c->g, 0x80);
    EXPECT_EQ(c->b, 0x00);
}

TEST(ColorParse, FromHexNoPrefixSixDigits) {
    auto c = SrgbU8::FromHex("ABCDEF");
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->r, 0xAB);
    EXPECT_EQ(c->g, 0xCD);
    EXPECT_EQ(c->b, 0xEF);
}

TEST(ColorParse, FromHex0xPrefixSixDigits) {
    auto c = SrgbU8::FromHex("0x123456");
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->r, 0x12);
    EXPECT_EQ(c->g, 0x34);
    EXPECT_EQ(c->b, 0x56);
}

TEST(ColorParse, FromHexShortFormThreeDigits) {
    auto c = SrgbU8::FromHex("#F80");
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->r, 0xFF);
    EXPECT_EQ(c->g, 0x88);
    EXPECT_EQ(c->b, 0x00);
}

TEST(ColorParse, FromHexLowercaseAccepted) {
    auto c = SrgbU8::FromHex("#abcdef");
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(c->r, 0xAB);
}

// ── 8-payload (RGBA) inputs MUST be rejected (point #4 strict contract) ─────

TEST(ColorParse, FromHexRejectsHashRGBA) { EXPECT_FALSE(SrgbU8::FromHex("#FF000080").has_value()); }

TEST(ColorParse, FromHexRejectsBareRGBA) { EXPECT_FALSE(SrgbU8::FromHex("FF000080").has_value()); }

TEST(ColorParse, FromHexRejectsHexPrefixedRGBA) {
    EXPECT_FALSE(SrgbU8::FromHex("0xFF000080").has_value());
}

// ── Other invalid inputs ────────────────────────────────────────────────────

TEST(ColorParse, FromHexRejectsEmpty) { EXPECT_FALSE(SrgbU8::FromHex("").has_value()); }

TEST(ColorParse, FromHexRejectsHashOnly) { EXPECT_FALSE(SrgbU8::FromHex("#").has_value()); }

TEST(ColorParse, FromHexRejectsNonHexChars) {
    EXPECT_FALSE(SrgbU8::FromHex("#GGGGGG").has_value());
    EXPECT_FALSE(SrgbU8::FromHex("#ZZZZZZ").has_value());
}

TEST(ColorParse, FromHexRejectsOddLengths) {
    EXPECT_FALSE(SrgbU8::FromHex("#1").has_value());
    EXPECT_FALSE(SrgbU8::FromHex("#12").has_value());
    EXPECT_FALSE(SrgbU8::FromHex("#1234").has_value());
    EXPECT_FALSE(SrgbU8::FromHex("#12345").has_value());
    EXPECT_FALSE(SrgbU8::FromHex("#1234567").has_value());
    EXPECT_FALSE(SrgbU8::FromHex("#123456789").has_value());
}

// ── ToHex format ────────────────────────────────────────────────────────────

TEST(ColorParse, ToHexUppercaseHashPrefix) {
    EXPECT_EQ(SrgbU8::ToHex(SrgbU8(0xFF, 0x80, 0x00)), "#FF8000");
    EXPECT_EQ(SrgbU8::ToHex(SrgbU8(0xAB, 0xCD, 0xEF)), "#ABCDEF");
}

TEST(ColorParse, FromToHexRoundTrip) {
    for (const std::string& hex : {"#000000", "#FFFFFF", "#FF8000", "#1A2B3C"}) {
        auto parsed = SrgbU8::FromHex(hex);
        ASSERT_TRUE(parsed.has_value()) << hex;
        EXPECT_EQ(SrgbU8::ToHex(*parsed), hex);
    }
}

// ── NamedColorLookup ────────────────────────────────────────────────────────

TEST(ColorParse, NamedColorLookupCanonical) {
    auto red = NamedColorLookup("red");
    ASSERT_TRUE(red.has_value());
    EXPECT_EQ(red->r, 255);
    EXPECT_EQ(red->g, 0);
    EXPECT_EQ(red->b, 0);
}

TEST(ColorParse, NamedColorLookupCaseAndPunctuation) {
    auto a = NamedColorLookup("Red");
    auto b = NamedColorLookup("RED");
    auto c = NamedColorLookup("re-d");
    ASSERT_TRUE(a.has_value());
    ASSERT_TRUE(b.has_value());
    ASSERT_TRUE(c.has_value());
    EXPECT_EQ(*a, *b);
    EXPECT_EQ(*a, *c);
}

TEST(ColorParse, NamedColorLookupUnknownReturnsNullopt) {
    EXPECT_FALSE(NamedColorLookup("zwxsq").has_value());
}

// ── HashFallbackColor: stable ───────────────────────────────────────────────

TEST(ColorParse, HashFallbackStableForSameKey) {
    SrgbU8 a = HashFallbackColor("strange-key");
    SrgbU8 b = HashFallbackColor("strange-key");
    EXPECT_EQ(a, b);
}

// ── FallbackColorByIndex: wraps ─────────────────────────────────────────────

TEST(ColorParse, FallbackColorByIndexWrapsModulo) {
    SrgbU8 zero = FallbackColorByIndex(0);
    SrgbU8 wrap = FallbackColorByIndex(16);
    EXPECT_EQ(zero, wrap);
}

TEST(ColorParse, FallbackColorByIndexNegativeWraps) {
    SrgbU8 zero    = FallbackColorByIndex(0);
    SrgbU8 neg_one = FallbackColorByIndex(-16);
    EXPECT_EQ(zero, neg_one);
}

// ── ResolveColorLiteral: 5-step resolution ──────────────────────────────────

TEST(ColorParse, ResolveColorLiteralUsesHexFirst) {
    SrgbU8 r = ResolveColorLiteral("#FF8000");
    EXPECT_EQ(r, SrgbU8(0xFF, 0x80, 0x00));
}

TEST(ColorParse, ResolveColorLiteralEmptyToMidGrey) {
    SrgbU8 r = ResolveColorLiteral("");
    EXPECT_EQ(r, SrgbU8(127, 127, 127));
}

TEST(ColorParse, ResolveColorLiteralNamedExact) {
    SrgbU8 r = ResolveColorLiteral("Yellow");
    EXPECT_EQ(r, SrgbU8(255, 255, 0));
}

TEST(ColorParse, ResolveColorLiteralFuzzyMatch) {
    // Contains "blue" substring → maps to blue.
    SrgbU8 r = ResolveColorLiteral("Blueberry-ish");
    EXPECT_EQ(r, SrgbU8(0, 0, 255));
}

TEST(ColorParse, ResolveColorLiteralUnknownGetsHashFallback) {
    SrgbU8 r        = ResolveColorLiteral("totally-unknown-color");
    SrgbU8 expected = HashFallbackColor("totallyunknowncolor");
    EXPECT_EQ(r, expected);
}
