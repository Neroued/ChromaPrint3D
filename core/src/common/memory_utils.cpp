/// \file memory_utils.cpp
/// \brief Cross-platform heap introspection and memory-return utilities.

#include "chromaprint3d/memory_utils.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <limits>

#if CHROMAPRINT3D_HAS_JEMALLOC
#    include <jemalloc/jemalloc.h>
#endif

#if defined(__linux__)
#    include <malloc.h>
#    include <unistd.h>
#elif defined(__APPLE__)
#    include <mach/mach.h>
#    include <sys/sysctl.h>
#elif defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    include <windows.h>
// must come after windows.h
#    include <psapi.h>
#endif

namespace ChromaPrint3D {

namespace {

bool IsUnlimitedMemoryValue(unsigned long long value) {
    return value == 0 || value >= (1ULL << 60);
}

#if defined(__linux__)
std::size_t ReadMemoryLimitFile(const char* path) {
    std::FILE* f = std::fopen(path, "r");
    if (!f) return 0;

    char buf[64]{};
    if (!std::fgets(buf, sizeof(buf), f)) {
        std::fclose(f);
        return 0;
    }
    std::fclose(f);

    if (std::strncmp(buf, "max", 3) == 0) { return 0; }

    char* end                = nullptr;
    const auto parsed_value  = std::strtoull(buf, &end, 10);
    const bool parsed_number = end != buf;
    if (!parsed_number || IsUnlimitedMemoryValue(parsed_value)) { return 0; }
    if (parsed_value > static_cast<unsigned long long>(std::numeric_limits<std::size_t>::max())) {
        return 0;
    }
    return static_cast<std::size_t>(parsed_value);
}
#endif

#if CHROMAPRINT3D_HAS_JEMALLOC
bool PurgeJemallocArenas() {
    unsigned narenas = 0;
    std::size_t sz   = sizeof(narenas);
    if (mallctl("arenas.narenas", &narenas, &sz, nullptr, 0) != 0 || narenas == 0) { return false; }

    bool purged_any = false;
    for (unsigned arena_idx = 0; arena_idx < narenas; ++arena_idx) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "arena.%u.purge", arena_idx);
        if (mallctl(buf, nullptr, nullptr, nullptr, 0) == 0) { purged_any = true; }
    }
    return purged_any;
}
#elif defined(__GLIBC__)
bool PurgeGlibcHeap() { return malloc_trim(0) != 0; }
#endif

} // namespace

// ---------------------------------------------------------------------------
// ReleaseFreedMemory — lock-free throttled purge
// ---------------------------------------------------------------------------

bool ReleaseFreedMemory(int min_interval_ms) {
    using namespace std::chrono;
    static std::atomic<int64_t> last_successful_purge_ms{0};
    static std::atomic<bool> purge_in_progress{false};

    auto now_ms = duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
    auto prev   = last_successful_purge_ms.load(std::memory_order_relaxed);
    if (now_ms - prev < min_interval_ms) return false;

    bool expected = false;
    if (!purge_in_progress.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                                   std::memory_order_relaxed)) {
        return false;
    }

    struct PurgeGuard {
        std::atomic<bool>& flag;

        ~PurgeGuard() { flag.store(false, std::memory_order_release); }
    } guard{purge_in_progress};

    prev = last_successful_purge_ms.load(std::memory_order_acquire);
    if (now_ms - prev < min_interval_ms) { return false; }

    bool purged = false;

#if CHROMAPRINT3D_HAS_JEMALLOC
    purged = PurgeJemallocArenas();
#elif defined(__GLIBC__)
    purged = PurgeGlibcHeap();
#endif

    if (purged) { last_successful_purge_ms.store(now_ms, std::memory_order_release); }
    return purged;
}

// ---------------------------------------------------------------------------
// GetProcessRssBytes
// ---------------------------------------------------------------------------

std::size_t GetProcessRssBytes() {
#if defined(__linux__)
    // /proc/self/statm fields: size resident shared text lib data dt (in pages)
    std::FILE* f = std::fopen("/proc/self/statm", "r");
    if (!f) return 0;
    unsigned long pages = 0;
    // skip first field (total vm size)
    if (std::fscanf(f, "%*lu %lu", &pages) != 1) pages = 0;
    std::fclose(f);
    return static_cast<std::size_t>(pages) * static_cast<std::size_t>(sysconf(_SC_PAGESIZE));
#elif defined(__APPLE__)
    mach_task_basic_info_data_t info{};
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, reinterpret_cast<task_info_t>(&info),
                  &count) == KERN_SUCCESS) {
        return static_cast<std::size_t>(info.resident_size);
    }
    return 0;
#elif defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return static_cast<std::size_t>(pmc.WorkingSetSize);
    }
    return 0;
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// GetMemoryLimitBytes
// ---------------------------------------------------------------------------

std::size_t GetMemoryLimitBytes() {
#if defined(__linux__)
    if (const auto v2_limit = ReadMemoryLimitFile("/sys/fs/cgroup/memory.max"); v2_limit > 0) {
        return v2_limit;
    }
    if (const auto v1_limit = ReadMemoryLimitFile("/sys/fs/cgroup/memory/memory.limit_in_bytes");
        v1_limit > 0) {
        return v1_limit;
    }
    long pages     = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0)
        return static_cast<std::size_t>(pages) * static_cast<std::size_t>(page_size);
    return 0;
#elif defined(__APPLE__)
    int mib[2]      = {CTL_HW, HW_MEMSIZE};
    uint64_t mem    = 0;
    std::size_t len = sizeof(mem);
    if (sysctl(mib, 2, &mem, &len, nullptr, 0) == 0) return static_cast<std::size_t>(mem);
    return 0;
#elif defined(_WIN32)
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms)) return static_cast<std::size_t>(ms.ullTotalPhys);
    return 0;
#else
    return 0;
#endif
}

// ---------------------------------------------------------------------------
// GetHeapStats
// ---------------------------------------------------------------------------

HeapStats GetHeapStats() {
    HeapStats st{};
#if CHROMAPRINT3D_HAS_JEMALLOC
    std::size_t epoch    = 1;
    std::size_t sz_epoch = sizeof(epoch);
    mallctl("epoch", &epoch, &sz_epoch, &epoch, sz_epoch);

    std::size_t sz = sizeof(std::size_t);
    mallctl("stats.allocated", &st.allocated, &sz, nullptr, 0);
    mallctl("stats.resident", &st.resident, &sz, nullptr, 0);
    mallctl("stats.mapped", &st.mapped, &sz, nullptr, 0);
    st.valid = true;
#elif defined(__GLIBC__)
#    if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 33)
    auto mi = mallinfo2();
#    else
    auto mi = mallinfo();
#    endif
    st.allocated = static_cast<std::size_t>(mi.uordblks);
    st.resident  = static_cast<std::size_t>(mi.arena + mi.hblkhd);
    st.mapped    = st.resident;
    st.valid     = true;
#endif
    return st;
}

} // namespace ChromaPrint3D
