#include "chromaprint3d/memory_utils.h"

#include <gtest/gtest.h>

#include <cstring>
#include <thread>
#include <vector>

namespace {

TEST(MemoryUtils, AllocatorNameIsNonEmpty) {
    const char* name = ChromaPrint3D::AllocatorName();
    ASSERT_NE(name, nullptr);
    EXPECT_GT(std::strlen(name), 0u);
}

TEST(MemoryUtils, GetProcessRssBytesReturnsNonZero) {
    std::size_t rss = ChromaPrint3D::GetProcessRssBytes();
    EXPECT_GT(rss, 0u);
}

TEST(MemoryUtils, GetMemoryLimitBytesReturnsNonZero) {
    std::size_t limit = ChromaPrint3D::GetMemoryLimitBytes();
    EXPECT_GT(limit, 0u);
}

TEST(MemoryUtils, GetHeapStatsReturnsValidResult) {
    auto stats = ChromaPrint3D::GetHeapStats();
    if (stats.valid) {
        EXPECT_GT(stats.allocated, 0u);
    } else {
        EXPECT_EQ(stats.allocated, 0u);
        EXPECT_EQ(stats.resident, 0u);
        EXPECT_EQ(stats.mapped, 0u);
    }
}

TEST(MemoryUtils, ReleaseFreedMemoryThrottlesCorrectly) {
    bool first  = ChromaPrint3D::ReleaseFreedMemory(0);
    bool second = ChromaPrint3D::ReleaseFreedMemory(60000);
    if (first) {
        EXPECT_FALSE(second);
    } else {
        EXPECT_FALSE(second);
    }
}

TEST(MemoryUtils, ReleaseFreedMemoryConcurrentCAS) {
    int success_count = 0;
    std::mutex mtx;
    std::vector<std::thread> threads;

    ChromaPrint3D::ReleaseFreedMemory(0);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&] {
            bool ok = ChromaPrint3D::ReleaseFreedMemory(0);
            if (ok) {
                std::lock_guard<std::mutex> lock(mtx);
                ++success_count;
            }
        });
    }
    for (auto& t : threads) t.join();

    EXPECT_LE(success_count, 1);
}

} // namespace
