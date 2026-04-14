/// \file memory_utils.h
/// \brief Cross-platform heap introspection and memory-return utilities.

#pragma once

#include <cstddef>
#include <cstdint>

namespace ChromaPrint3D {

/// Return freed heap pages to the OS.
/// Built-in throttle: skips if the last successful purge was less than
/// \p min_interval_ms ago (default 5 000 ms). Thread-safe.
/// \return true only if memory was actually returned to the OS.
bool ReleaseFreedMemory(int min_interval_ms = 5000);

/// Current process RSS in bytes (0 on failure).
std::size_t GetProcessRssBytes();

/// Container / system memory limit in bytes (0 on failure).
/// Linux: cgroup v2 `/sys/fs/cgroup/memory.max`, then cgroup v1
/// `/sys/fs/cgroup/memory/memory.limit_in_bytes`, then total physical RAM.
/// Other platforms: total physical RAM.
std::size_t GetMemoryLimitBytes();

struct HeapStats {
    std::size_t allocated = 0; // application-level allocated
    std::size_t resident  = 0; // physical memory held by allocator
    std::size_t mapped    = 0; // virtual memory mapped by allocator
    bool valid            = false;
};

HeapStats GetHeapStats();

/// Compile-time allocator identifier.
constexpr const char* AllocatorName() {
#if CHROMAPRINT3D_HAS_JEMALLOC
    return "jemalloc";
#elif defined(__GLIBC__)
    return "glibc";
#else
    return "system";
#endif
}

} // namespace ChromaPrint3D
