#include <gtest/gtest.h>
#include "chromaprint3d/kdtree.h"
#include "chromaprint3d/color.h"

#include <cstddef>
#include <vector>

using namespace ChromaPrint3D;

struct LabProjection {
    const Lab& operator()(const Lab& lab) const { return lab; }
};

using TestTree = kdt::KDTree<Lab, 3, LabProjection, std::size_t, float>;

TEST(KDTree, NearestSinglePoint) {
    std::vector<Lab> points = {Lab(50.0f, 0.0f, 0.0f)};
    std::vector<std::size_t> indices = {0};
    TestTree tree;
    tree.Reset(points, indices, LabProjection{});

    auto result = tree.Nearest(Lab(51.0f, 0.0f, 0.0f));
    EXPECT_EQ(result.index, 0u);
    EXPECT_NEAR(result.dist2, 1.0f, 1e-5f);
}

TEST(KDTree, NearestCorrectness) {
    std::vector<Lab> points = {
        Lab(0.0f, 0.0f, 0.0f),
        Lab(50.0f, 0.0f, 0.0f),
        Lab(100.0f, 0.0f, 0.0f),
    };
    std::vector<std::size_t> indices = {0, 1, 2};
    TestTree tree;
    tree.Reset(points, indices, LabProjection{});

    auto r0 = tree.Nearest(Lab(10.0f, 0.0f, 0.0f));
    EXPECT_EQ(r0.index, 0u);

    auto r1 = tree.Nearest(Lab(48.0f, 0.0f, 0.0f));
    EXPECT_EQ(r1.index, 1u);

    auto r2 = tree.Nearest(Lab(90.0f, 0.0f, 0.0f));
    EXPECT_EQ(r2.index, 2u);
}

TEST(KDTree, KNearestNeighbors) {
    std::vector<Lab> points;
    std::vector<std::size_t> indices;
    for (int i = 0; i < 20; ++i) {
        points.emplace_back(static_cast<float>(i * 5), 0.0f, 0.0f);
        indices.push_back(static_cast<std::size_t>(i));
    }
    TestTree tree;
    tree.Reset(points, indices, LabProjection{});

    std::vector<kdt::Neighbor<std::size_t, float>> neighbors;
    tree.KNearest(Lab(52.0f, 0.0f, 0.0f), 3, neighbors);
    EXPECT_EQ(neighbors.size(), 3u);

    bool found_closest = false;
    for (const auto& n : neighbors) {
        if (n.index == 10) { found_closest = true; }
    }
    EXPECT_TRUE(found_closest);
}

TEST(KDTree, ExactMatch) {
    std::vector<Lab> points = {Lab(42.0f, 10.0f, -5.0f)};
    std::vector<std::size_t> indices = {0};
    TestTree tree;
    tree.Reset(points, indices, LabProjection{});

    auto result = tree.Nearest(Lab(42.0f, 10.0f, -5.0f));
    EXPECT_EQ(result.index, 0u);
    EXPECT_NEAR(result.dist2, 0.0f, 1e-10f);
}
