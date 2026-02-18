#include <gtest/gtest.h>
#include "chromaprint3d/voxel.h"

using namespace ChromaPrint3D;

TEST(VoxelGrid, GetSetBasic) {
    VoxelGrid grid;
    grid.width      = 4;
    grid.height     = 4;
    grid.num_layers = 3;
    grid.ooc.assign(static_cast<size_t>(4 * 4 * 3), 0);

    EXPECT_FALSE(grid.Get(0, 0, 0));

    grid.Set(2, 1, 1, true);
    EXPECT_TRUE(grid.Get(2, 1, 1));
    EXPECT_FALSE(grid.Get(2, 1, 0));

    grid.Set(2, 1, 1, false);
    EXPECT_FALSE(grid.Get(2, 1, 1));
}

TEST(VoxelGrid, OutOfBoundsReturnsFlase) {
    VoxelGrid grid;
    grid.width      = 2;
    grid.height     = 2;
    grid.num_layers = 2;
    grid.ooc.assign(static_cast<size_t>(2 * 2 * 2), 0);

    EXPECT_FALSE(grid.Get(-1, 0, 0));
    EXPECT_FALSE(grid.Get(0, -1, 0));
    EXPECT_FALSE(grid.Get(0, 0, -1));
    EXPECT_FALSE(grid.Get(3, 0, 0));
}

TEST(VoxelGrid, SetAllVoxels) {
    VoxelGrid grid;
    grid.width      = 3;
    grid.height     = 3;
    grid.num_layers = 2;
    grid.ooc.assign(static_cast<size_t>(3 * 3 * 2), 0);

    for (int w = 0; w < 3; ++w) {
        for (int h = 0; h < 3; ++h) {
            for (int l = 0; l < 2; ++l) {
                grid.Set(w, h, l, true);
            }
        }
    }

    for (int w = 0; w < 3; ++w) {
        for (int h = 0; h < 3; ++h) {
            for (int l = 0; l < 2; ++l) {
                EXPECT_TRUE(grid.Get(w, h, l));
            }
        }
    }
}
