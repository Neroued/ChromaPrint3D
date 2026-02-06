#pragma once

#include <algorithm>
#include <concepts>
#include <limits>
#include <queue>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

/* 针对点集的非持有型 KDTree
 *
 *
 */

namespace kdt {

// 成员访问需求： coord(i) 或者 [i]
template <class P>
concept HasMemberCoord = requires(const P& p, int i) { p.Coord(i); };

template <class P>
concept HasSubscript = requires(const P& p, int i) { p[i]; };

template <class P>
    requires HasMemberCoord<P> || HasSubscript<P>
auto Coord(const P& p, int i) {
    if constexpr (HasMemberCoord<P>) {
        return p.Coord(i);
    } else {
        return p[i];
    }
}

template <class P>
using CoordT = std::remove_cvref_t<decltype(Coord<P>(std::declval<const P&>(), 0))>;

template <class P, int K>
concept CoordAccessibleK =
    (K > 0) && (HasMemberCoord<P> || HasSubscript<P>) && std::is_arithmetic_v<CoordT<P>>;

// 距离函数需求： P::Dist2() 或 ::Dist2()，否则使用默认
template <class P>
concept HasStaticDist2 = requires(const P& a, const P& b) {
    { P::Dist2(a, b) } -> std::convertible_to<double>;
};

namespace detail {
template <class P>
concept HasADLDist2 = requires(const P& a, const P& b) {
    { Dist2(a, b) } -> std::convertible_to<double>;
};

// 避免函数同名，包装一层
template <class P>
auto CallADLDist2(const P& a, const P& b) {
    return Dist2(a, b);
}
} // namespace detail

template <class P, int K>
    requires CoordAccessibleK<P, K>
auto DefaultDist2(const P& a, const P& b) {
    using AccT = std::common_type_t<CoordT<P>, double>;
    AccT sum   = AccT(0);
#pragma unroll
    for (int i = 0; i < K; ++i) {
        AccT x = static_cast<AccT>(Coord<P>(a, i)) - static_cast<AccT>(Coord<P>(b, i));
        sum += x * x;
    }
    return sum;
}

template <class P, int K>
    requires CoordAccessibleK<P, K>
auto Dist2(const P& a, const P& b) {
    if constexpr (HasStaticDist2<P>) {
        return P::Dist2(a, b);
    } else if constexpr (detail::HasADLDist2<P>) {
        return detail::CallADLDist2(a, b);
    } else {
        return DefaultDist2<P, K>(a, b);
    }
}

template <class Index, class Scalar>
struct Neighbor {
    Index index;  // 指向原始点集 points[index]
    Scalar dist2; // 到 query 的平方距离
};

template <class P, int K, class Proj, class Index = int, class Scalar = double>
    requires CoordAccessibleK<std::remove_cvref_t<std::invoke_result_t<Proj&, const P&>>, K>
class KDTree {
public:
    using ProjPoint  = std::remove_cvref_t<std::invoke_result_t<Proj&, const P&>>;
    using CoordType  = CoordT<ProjPoint>;
    using NeighborT  = Neighbor<Index, Scalar>;
    using IndexSpan  = std::span<const Index>;
    using PointsSpan = std::span<const P>;

    KDTree() = default;

    KDTree(PointsSpan points, IndexSpan indices, Proj proj = {})
        : points_(points), proj_(std::move(proj)) {
        BuildFrom(indices);
    }

    void Reset(PointsSpan points, IndexSpan indices, Proj proj = {}) {
        points_ = points;
        proj_   = std::move(proj);
        BuildFrom(indices);
    }

    [[nodiscard]] bool Empty() const { return order_.empty(); }

    [[nodiscard]] std::size_t Size() const { return order_.size(); }

    [[nodiscard]] PointsSpan Points() const { return points_; }

    [[nodiscard]] IndexSpan Indices() const { return IndexSpan(order_); }

    NeighborT Nearest(const ProjPoint& query) const {
        NeighborT best{Index{}, std::numeric_limits<Scalar>::infinity()};
        if (root_ < 0) { return best; }
        NearestImpl(root_, query, best);
        return best;
    }

    NeighborT NearestFrom(const P& query) const { return Nearest(proj_(query)); }

    void RadiusSearch(const ProjPoint& query, Scalar radius, std::vector<NeighborT>& out) const {
        out.clear();
        if (root_ < 0) { return; }
        const Scalar r2 = radius * radius;
        RadiusImpl(root_, query, r2, out);
    }

    void RadiusSearchFrom(const P& query, Scalar radius, std::vector<NeighborT>& out) const {
        RadiusSearch(proj_(query), radius, out);
    }

    void KNearest(const ProjPoint& query, std::size_t k, std::vector<NeighborT>& out) const {
        out.clear();
        if (k == 0 || root_ < 0) { return; }
        Heap heap;
        KNearestImpl(root_, query, k, heap);
        out.reserve(heap.size());
        while (!heap.empty()) {
            out.push_back(heap.top());
            heap.pop();
        }
        std::sort(out.begin(), out.end(),
                  [](const NeighborT& a, const NeighborT& b) { return a.dist2 < b.dist2; });
    }

    void KNearestFrom(const P& query, std::size_t k, std::vector<NeighborT>& out) const {
        KNearest(proj_(query), k, out);
    }

private:
    struct Node {
        Index index;
        int left;
        int right;
        int axis;
    };

    struct MaxDist {
        bool operator()(const NeighborT& a, const NeighborT& b) const { return a.dist2 < b.dist2; }
    };

    using Heap = std::priority_queue<NeighborT, std::vector<NeighborT>, MaxDist>;

    using AccT = std::common_type_t<CoordType, Scalar>;

    void BuildFrom(IndexSpan indices) {
        order_.assign(indices.begin(), indices.end());
        nodes_.clear();
        nodes_.reserve(order_.size());
        root_ = Build(0, order_.size(), 0);
    }

    int Build(std::size_t begin, std::size_t end, int depth) {
        if (begin >= end) { return -1; }
        const std::size_t mid = begin + (end - begin) / 2;
        const int axis        = depth % K;
        auto comp             = [this, axis](Index a, Index b) {
            return CoordAt(ProjectPoint(a), axis) < CoordAt(ProjectPoint(b), axis);
        };
        std::nth_element(order_.begin() + begin, order_.begin() + mid, order_.begin() + end, comp);

        const int node_index = static_cast<int>(nodes_.size());
        nodes_.push_back(Node{order_[mid], -1, -1, axis});
        nodes_[node_index].left  = Build(begin, mid, depth + 1);
        nodes_[node_index].right = Build(mid + 1, end, depth + 1);
        return node_index;
    }

    static AccT CoordAt(const ProjPoint& p, int axis) {
        return static_cast<AccT>(Coord<ProjPoint>(p, axis));
    }

    const P& At(Index index) const { return points_[static_cast<std::size_t>(index)]; }

    ProjPoint ProjectPoint(Index index) const { return proj_(At(index)); }

    void NearestImpl(int node_index, const ProjPoint& query, NeighborT& best) const {
        const Node& node = nodes_[node_index];
        const auto point = ProjectPoint(node.index);
        const Scalar d2  = static_cast<Scalar>(Dist2<ProjPoint, K>(query, point));
        if (d2 < best.dist2) { best = NeighborT{node.index, d2}; }

        const AccT diff  = CoordAt(query, node.axis) - CoordAt(point, node.axis);
        const AccT diff2 = diff * diff;

        const int near_child = diff <= AccT(0) ? node.left : node.right;
        const int far_child  = diff <= AccT(0) ? node.right : node.left;

        if (near_child >= 0) { NearestImpl(near_child, query, best); }
        if (far_child >= 0 && diff2 < static_cast<AccT>(best.dist2)) {
            NearestImpl(far_child, query, best);
        }
    }

    void RadiusImpl(int node_index, const ProjPoint& query, Scalar r2,
                    std::vector<NeighborT>& out) const {
        const Node& node = nodes_[node_index];
        const auto point = ProjectPoint(node.index);
        const Scalar d2  = static_cast<Scalar>(Dist2<ProjPoint, K>(query, point));
        if (d2 <= r2) { out.push_back(NeighborT{node.index, d2}); }

        const AccT diff  = CoordAt(query, node.axis) - CoordAt(point, node.axis);
        const AccT diff2 = diff * diff;

        const int near_child = diff <= AccT(0) ? node.left : node.right;
        const int far_child  = diff <= AccT(0) ? node.right : node.left;

        if (near_child >= 0) { RadiusImpl(near_child, query, r2, out); }
        if (far_child >= 0 && diff2 <= static_cast<AccT>(r2)) {
            RadiusImpl(far_child, query, r2, out);
        }
    }

    void KNearestImpl(int node_index, const ProjPoint& query, std::size_t k, Heap& heap) const {
        const Node& node = nodes_[node_index];
        const auto point = ProjectPoint(node.index);
        const Scalar d2  = static_cast<Scalar>(Dist2<ProjPoint, K>(query, point));
        if (heap.size() < k) {
            heap.push(NeighborT{node.index, d2});
        } else if (d2 < heap.top().dist2) {
            heap.pop();
            heap.push(NeighborT{node.index, d2});
        }

        const AccT diff  = CoordAt(query, node.axis) - CoordAt(point, node.axis);
        const AccT diff2 = diff * diff;

        const int near_child = diff <= AccT(0) ? node.left : node.right;
        const int far_child  = diff <= AccT(0) ? node.right : node.left;

        if (near_child >= 0) { KNearestImpl(near_child, query, k, heap); }

        const Scalar best =
            heap.size() < k ? std::numeric_limits<Scalar>::infinity() : heap.top().dist2;
        if (far_child >= 0 && diff2 <= static_cast<AccT>(best)) {
            KNearestImpl(far_child, query, k, heap);
        }
    }

    PointsSpan points_{};
    Proj proj_{};
    std::vector<Index> order_{};
    std::vector<Node> nodes_{};
    int root_ = -1;
};



} // namespace kdt