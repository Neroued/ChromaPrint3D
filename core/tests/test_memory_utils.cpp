#include "chromaprint3d/memory_utils.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstring>
#include <latch>
#include <mutex>
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
    ChromaPrint3D::ReleaseFreedMemory(0);
    EXPECT_FALSE(ChromaPrint3D::ReleaseFreedMemory(60000));
}

TEST(MemoryUtils, ReleaseFreedMemoryConcurrentCallsRespectThrottle) {
    constexpr int kThreadCount    = 8;
    constexpr int kMinIntervalMs  = 100;
    constexpr int kStartupSlackMs = 10;
    int success_count             = 0;
    std::mutex mtx;
    std::latch ready(kThreadCount);
    std::latch start(1);
    std::vector<std::thread> threads;

    // Let any prior successful purge age out so this test is independent
    // from the previous test's static throttle state.
    std::this_thread::sleep_for(std::chrono::milliseconds(kMinIntervalMs + kStartupSlackMs));

    for (int i = 0; i < kThreadCount; ++i) {
        threads.emplace_back([&] {
            ready.count_down();
            start.wait();

            bool ok = ChromaPrint3D::ReleaseFreedMemory(kMinIntervalMs);
            if (ok) {
                std::lock_guard<std::mutex> lock(mtx);
                ++success_count;
            }
        });
    }
    ready.wait();
    start.count_down();
    for (auto& t : threads) t.join();

    EXPECT_LE(success_count, 1);
}

} // namespace
