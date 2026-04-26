#pragma once

// Internal templated helpers for typed matcher dispatch.
// NOT part of the public API.
//
// The matcher core works in either project-D65 Lab or linear RGB; the choice
// is encoded in the type of the target color (`Lab` or `Rgb`) rather than a
// runtime `bool use_lab` flag. These helpers provide the small bits of
// space-dependent logic (target-to-Lab projection for reporting, scoring
// distance, and `cv::Vec3f` boundary unwrap) that recur across
// `candidate_select.cpp`, `dither.cpp`, and `vector_match.cpp`.

// `match_utils.h` already includes `chromaprint3d/color.h`, which transitively
// pulls in the strong `Lab` / `Rgb` types and the conversion helpers we use
// here (`Lab::FromRgb`, `candidate_lab.ToRgb()`).
#include "match_utils.h"

#include <opencv2/core.hpp>

#include <type_traits>

namespace ChromaPrint3D {
namespace detail {

/// Project a typed target color into Lab. Used to compute reporting metrics
/// such as `lab_dist2` regardless of the matching color space.
template <typename T>
inline Lab AsLab(const T& target) {
    static_assert(std::is_same_v<T, Lab> || std::is_same_v<T, Rgb>,
                  "AsLab supports only Lab or Rgb target types");
    if constexpr (std::is_same_v<T, Lab>) {
        return target;
    } else {
        return Lab::FromRgb(target);
    }
}

/// Squared distance used as the matcher score, in the matching color space.
/// `candidate_lab` is the Lab color of the candidate (always stored in Lab
/// form by ColorDB / ModelPackage); `target` is the typed user target.
template <typename T>
inline float ScoreDist2(const Lab& candidate_lab, const T& target) {
    static_assert(std::is_same_v<T, Lab> || std::is_same_v<T, Rgb>,
                  "ScoreDist2 supports only Lab or Rgb target types");
    if constexpr (std::is_same_v<T, Lab>) {
        return Dist2(candidate_lab, target);
    } else {
        return Dist2(candidate_lab.ToRgb(), target);
    }
}

/// Unwrap a `cv::Vec3f` pixel into the typed target color. The pixel is read
/// from the matching-space `cv::Mat` (`img.lab` or `img.rgb`), so the channel
/// order is well defined: for `Lab` the components are (L, a, b); for `Rgb`
/// they are (r, g, b) in linear sRGB.
template <typename T>
inline T MakeTarget(const cv::Vec3f& v) {
    static_assert(std::is_same_v<T, Lab> || std::is_same_v<T, Rgb>,
                  "MakeTarget supports only Lab or Rgb target types");
    return T(v[0], v[1], v[2]);
}

} // namespace detail
} // namespace ChromaPrint3D
