#include <gtest/gtest.h>
#include "chromaprint3d/vec3.h"

using namespace ChromaPrint3D;

TEST(Vec3i, DefaultConstruction) {
    Vec3i v;
    EXPECT_EQ(v.x, 0);
    EXPECT_EQ(v.y, 0);
    EXPECT_EQ(v.z, 0);
}

TEST(Vec3i, Addition) {
    Vec3i a(1, 2, 3), b(4, 5, 6);
    Vec3i c = a + b;
    EXPECT_EQ(c.x, 5);
    EXPECT_EQ(c.y, 7);
    EXPECT_EQ(c.z, 9);
}

TEST(Vec3i, Subtraction) {
    Vec3i a(10, 20, 30), b(3, 5, 7);
    Vec3i c = a - b;
    EXPECT_EQ(c.x, 7);
    EXPECT_EQ(c.y, 15);
    EXPECT_EQ(c.z, 23);
}

TEST(Vec3i, ScalarMultiply) {
    Vec3i v(2, 3, 4);
    Vec3i r = v * 3;
    EXPECT_EQ(r.x, 6);
    EXPECT_EQ(r.y, 9);
    EXPECT_EQ(r.z, 12);

    Vec3i l = 3 * v;
    EXPECT_EQ(l.x, 6);
}

TEST(Vec3i, DotProduct) {
    Vec3i a(1, 0, 0), b(0, 1, 0);
    EXPECT_EQ(a.Dot(b), 0);

    Vec3i c(1, 2, 3);
    EXPECT_EQ(c.Dot(c), 14);
}

TEST(Vec3i, Indexing) {
    Vec3i v(10, 20, 30);
    EXPECT_EQ(v[0], 10);
    EXPECT_EQ(v[1], 20);
    EXPECT_EQ(v[2], 30);
    v[1] = 99;
    EXPECT_EQ(v.y, 99);
}

TEST(Vec3f, DefaultConstruction) {
    Vec3f v;
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
    EXPECT_FLOAT_EQ(v.z, 0.0f);
}

TEST(Vec3f, LengthAndNormalize) {
    Vec3f v(3.0f, 4.0f, 0.0f);
    EXPECT_FLOAT_EQ(v.Length(), 5.0f);

    Vec3f n = v.Normalized();
    EXPECT_NEAR(n.Length(), 1.0f, 1e-6f);
}

TEST(Vec3f, Lerp) {
    Vec3f a(0.0f, 0.0f, 0.0f), b(10.0f, 10.0f, 10.0f);
    Vec3f mid = Vec3f::Lerp(a, b, 0.5f);
    EXPECT_NEAR(mid.x, 5.0f, 1e-6f);
    EXPECT_NEAR(mid.y, 5.0f, 1e-6f);
}

TEST(Vec3f, Clamp01) {
    Vec3f v(-0.5f, 0.5f, 1.5f);
    Vec3f c = Vec3f::Clamp01(v);
    EXPECT_FLOAT_EQ(c.x, 0.0f);
    EXPECT_FLOAT_EQ(c.y, 0.5f);
    EXPECT_FLOAT_EQ(c.z, 1.0f);
}

TEST(Vec3f, NearlyEqual) {
    Vec3f a(1.0f, 2.0f, 3.0f);
    Vec3f b(1.0f + 1e-7f, 2.0f, 3.0f);
    EXPECT_TRUE(a.NearlyEqual(b));

    Vec3f c(1.1f, 2.0f, 3.0f);
    EXPECT_FALSE(a.NearlyEqual(c));
}

TEST(Vec3f, Distance) {
    Vec3f a(0.0f, 0.0f, 0.0f), b(1.0f, 0.0f, 0.0f);
    EXPECT_FLOAT_EQ(Vec3f::Distance(a, b), 1.0f);
}

TEST(Vec3f, CompoundAssignment) {
    Vec3f v(1.0f, 2.0f, 3.0f);
    v += Vec3f(1.0f, 1.0f, 1.0f);
    EXPECT_FLOAT_EQ(v.x, 2.0f);
    v *= 2.0f;
    EXPECT_FLOAT_EQ(v.x, 4.0f);
}
