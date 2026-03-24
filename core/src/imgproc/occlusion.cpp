#include "occlusion.h"

#include <clipper2/clipper.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(_OPENMP)
#    if defined(__has_include)
#        if __has_include(<omp.h>)
#            include <omp.h>
#        endif
#    else
#        include <omp.h>
#    endif
#endif

namespace ChromaPrint3D::detail {

namespace {

constexpr double kScale = 1000000.0;

Clipper2Lib::Path64 ContourToPath(const Contour& c) {
    Clipper2Lib::Path64 path;
    path.reserve(c.size());
    for (const Vec2f& p : c) {
        path.emplace_back(static_cast<int64_t>(std::round(p.x * kScale)),
                          static_cast<int64_t>(std::round(p.y * kScale)));
    }
    return path;
}

Contour PathToContour(const Clipper2Lib::Path64& path) {
    Contour c;
    c.reserve(path.size());
    for (const auto& pt : path) {
        c.emplace_back(static_cast<float>(pt.x) / static_cast<float>(kScale),
                       static_cast<float>(pt.y) / static_cast<float>(kScale));
    }
    return c;
}

Clipper2Lib::Paths64 ShapeToPaths(const VectorShape& shape) {
    Clipper2Lib::Paths64 paths;
    paths.reserve(shape.contours.size());
    for (const Contour& c : shape.contours) {
        if (c.size() >= 3) { paths.push_back(ContourToPath(c)); }
    }
    return paths;
}

VectorShape PathsToShape(const Clipper2Lib::Paths64& paths, const VectorShape& original) {
    VectorShape result;
    result.fill_type     = original.fill_type;
    result.svg_fill_rule = original.svg_fill_rule;
    result.fill_color    = original.fill_color;
    result.gradient      = original.gradient;
    result.opacity       = original.opacity;
    result.contours.reserve(paths.size());
    for (const auto& path : paths) {
        if (path.size() >= 3) { result.contours.push_back(PathToContour(path)); }
    }
    return result;
}

Clipper2Lib::FillRule ToClipperFillRule(SvgFillRule rule) {
    return (rule == SvgFillRule::EvenOdd) ? Clipper2Lib::FillRule::EvenOdd
                                          : Clipper2Lib::FillRule::NonZero;
}

struct AABB {
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float max_y = std::numeric_limits<float>::lowest();

    bool Valid() const { return min_x <= max_x && min_y <= max_y; }

    bool Overlaps(const AABB& o) const {
        return min_x <= o.max_x && max_x >= o.min_x && min_y <= o.max_y && max_y >= o.min_y;
    }
};

AABB ComputeAABB(const VectorShape& shape) {
    AABB bb;
    for (const auto& contour : shape.contours) {
        for (const Vec2f& p : contour) {
            bb.min_x = std::min(bb.min_x, p.x);
            bb.min_y = std::min(bb.min_y, p.y);
            bb.max_x = std::max(bb.max_x, p.x);
            bb.max_y = std::max(bb.max_y, p.y);
        }
    }
    return bb;
}

struct SpatialGrid {
    int cols       = 0;
    int rows       = 0;
    float cell_w   = 0.0f;
    float cell_h   = 0.0f;
    float origin_x = 0.0f;
    float origin_y = 0.0f;
    std::vector<std::vector<int>> cells;

    void Build(const std::vector<AABB>& bboxes, int n) {
        AABB world;
        for (int i = 0; i < n; ++i) {
            if (!bboxes[static_cast<size_t>(i)].Valid()) continue;
            world.min_x = std::min(world.min_x, bboxes[static_cast<size_t>(i)].min_x);
            world.min_y = std::min(world.min_y, bboxes[static_cast<size_t>(i)].min_y);
            world.max_x = std::max(world.max_x, bboxes[static_cast<size_t>(i)].max_x);
            world.max_y = std::max(world.max_y, bboxes[static_cast<size_t>(i)].max_y);
        }
        if (!world.Valid()) return;

        int dim  = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(n) / 4.0)));
        cols     = dim;
        rows     = dim;
        origin_x = world.min_x;
        origin_y = world.min_y;
        float w  = world.max_x - world.min_x;
        float h  = world.max_y - world.min_y;
        cell_w   = (w > 0.0f) ? w / static_cast<float>(cols) : 1.0f;
        cell_h   = (h > 0.0f) ? h / static_cast<float>(rows) : 1.0f;
        cells.resize(static_cast<size_t>(cols * rows));

        for (int i = 0; i < n; ++i) {
            const AABB& bb = bboxes[static_cast<size_t>(i)];
            if (!bb.Valid()) continue;
            int x0 = CellX(bb.min_x);
            int y0 = CellY(bb.min_y);
            int x1 = CellX(bb.max_x);
            int y1 = CellY(bb.max_y);
            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    cells[static_cast<size_t>(y * cols + x)].push_back(i);
                }
            }
        }
    }

    int CellX(float x) const {
        int cx = static_cast<int>((x - origin_x) / cell_w);
        return std::clamp(cx, 0, cols - 1);
    }

    int CellY(float y) const {
        int cy = static_cast<int>((y - origin_y) / cell_h);
        return std::clamp(cy, 0, rows - 1);
    }
};

} // namespace

std::vector<VectorShape> ClipOcclusion(const std::vector<VectorShape>& shapes) {
    if (shapes.empty()) { return {}; }

    const int n  = static_cast<int>(shapes.size());
    auto t_start = std::chrono::steady_clock::now();

    std::vector<AABB> bboxes(static_cast<size_t>(n));
    std::vector<Clipper2Lib::Paths64> all_paths(static_cast<size_t>(n));

    for (int i = 0; i < n; ++i) {
        bboxes[static_cast<size_t>(i)]    = ComputeAABB(shapes[static_cast<size_t>(i)]);
        all_paths[static_cast<size_t>(i)] = ShapeToPaths(shapes[static_cast<size_t>(i)]);
    }

    auto t_prep = std::chrono::steady_clock::now();

    SpatialGrid grid;
    grid.Build(bboxes, n);

    auto t_grid = std::chrono::steady_clock::now();

    spdlog::info("[perf] ClipOcclusion prep: bbox+paths={:.1f} ms, grid={}x{} built in {:.1f} ms",
                 std::chrono::duration<double, std::milli>(t_prep - t_start).count(), grid.cols,
                 grid.rows, std::chrono::duration<double, std::milli>(t_grid - t_prep).count());

    std::vector<VectorShape> result(static_cast<size_t>(n));
    const std::ptrdiff_t pn = static_cast<std::ptrdiff_t>(n);

#if defined(_OPENMP)
#    pragma omp parallel
#endif
    {
        std::vector<int> candidates;
        candidates.reserve(64);

#if defined(_OPENMP)
#    pragma omp for schedule(dynamic)
#endif
        for (std::ptrdiff_t pi = 0; pi < pn; ++pi) {
            const int i          = static_cast<int>(pi);
            const size_t idx     = static_cast<size_t>(i);
            const AABB& bb       = bboxes[idx];
            const auto& my_paths = all_paths[idx];
            if (my_paths.empty() || !bb.Valid()) continue;

            Clipper2Lib::FillRule fr = ToClipperFillRule(shapes[idx].svg_fill_rule);

            int x0 = grid.CellX(bb.min_x);
            int y0 = grid.CellY(bb.min_y);
            int x1 = grid.CellX(bb.max_x);
            int y1 = grid.CellY(bb.max_y);

            candidates.clear();
            for (int cy = y0; cy <= y1; ++cy) {
                for (int cx = x0; cx <= x1; ++cx) {
                    for (int j : grid.cells[static_cast<size_t>(cy * grid.cols + cx)]) {
                        if (j > i) { candidates.push_back(j); }
                    }
                }
            }

            Clipper2Lib::Paths64 clip_paths;
            if (!candidates.empty()) {
                std::sort(candidates.begin(), candidates.end());
                candidates.erase(std::unique(candidates.begin(), candidates.end()),
                                 candidates.end());
                for (int j : candidates) {
                    if (!bb.Overlaps(bboxes[static_cast<size_t>(j)])) continue;
                    for (const auto& p : all_paths[static_cast<size_t>(j)]) {
                        clip_paths.push_back(p);
                    }
                }
            }

            Clipper2Lib::Paths64 clipped;
            if (clip_paths.empty()) {
                clipped = Clipper2Lib::Union(my_paths, fr);
            } else {
                clipped = Clipper2Lib::Difference(my_paths, clip_paths, fr);
            }

            if (!clipped.empty()) { result[idx] = PathsToShape(clipped, shapes[idx]); }
        }
    }

    std::vector<VectorShape> final_result;
    final_result.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        if (!result[static_cast<size_t>(i)].contours.empty()) {
            final_result.push_back(std::move(result[static_cast<size_t>(i)]));
        }
    }

    double total_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start)
            .count();
    spdlog::info("[perf] ClipOcclusion done: {} -> {} shapes in {:.1f} ms (grid {}x{})", n,
                 final_result.size(), total_ms, grid.cols, grid.rows);
    return final_result;
}

} // namespace ChromaPrint3D::detail
