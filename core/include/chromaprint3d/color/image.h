#pragma once

/// \file color/image.h
/// \brief `cv::Mat`-level color conversion entries. The **only** sub-header
///        in `chromaprint3d/color/` that depends on OpenCV.
///
/// Authoritative `BgrToLab` self-implements the full sRGB → linear → XYZ(D65)
/// → Lab(D65) pipeline using the project's analytical D65 math. **Forbidden**
/// to call `cv::cvtColor(BGR2Lab)` from inside this entry — the OpenCV LUT
/// path produces sub-ΔE drift incompatible with our `Lab::DeltaE76` thresholds.
///
/// Channel-order contract: OpenCV inputs are BGR; the project's standard
/// sRGB→XYZ matrix is RGB-ordered. We use a pre-computed BGR-column matrix
/// (`kBgrToXyzD65Matrix`, columns `(B, G, R)`) so `cv::transform` does both
/// the channel reorder and matrix multiply in a single pass. **Never** apply
/// the standard RGB→XYZ matrix directly to BGR pixel data.

#include "chromaprint3d/color/types.h"

#include <opencv2/core.hpp>

namespace ChromaPrint3D {

// ── Image-level entries (defined in core/src/color/image.cpp) ───────────────
//
// Per-pixel `cv::Vec3f` ↔ `Rgb` / `Lab` conversion helpers were intentionally
// removed: their channel-order semantics were ambiguous (BGR vs RGB depending
// on which `cv::Mat` the pixel came from). Call sites should construct the
// strong type explicitly at the OpenCV boundary, e.g.
//   `Rgb r(v[0], v[1], v[2]);` or `Lab lab(v[0], v[1], v[2]);`
// where the channel order of `v` is locally known.

/// 8UC3 BGR (gamma-encoded, OpenCV convention) → 32FC3 Lab (project D65).
///
/// Input: `cv::Mat` with `type() == CV_8UC3`, channel order BGR.
/// Output: `cv::Mat` with `type() == CV_32FC3`, channel order (L, a, b).
///
/// Internal pipeline:
///   1. `cv::LUT` per-channel: BGR uint8 → BGR linear float [0,1] (256-entry LUT).
///   2. `cv::transform` with `kBgrToXyzD65Matrix` (columns: B, G, R): one-pass
///      channel reorder + RGB→XYZ matrix multiply.
///   3. `cv::parallel_for_` row-major loop applying scalar `LabF` (with 4096-
///      entry LUT acceleration) and the standard L\*a\*b\* affine.
///
/// Empty input → empty output.
cv::Mat BgrToLab(const cv::Mat& bgr_u8);

/// 8UC3 BGR → 32FC3 linear RGB (RGB channel order, gamma-decoded).
/// Replaces `raster_proc.cpp::BgrToRgbLinear`. Channel-order suffix `RgbLinear`
/// makes the post-conversion order explicit.
cv::Mat BgrToLinearRgb(const cv::Mat& bgr_u8);

/// 32FC3 linear RGB → 8UC3 BGR (gamma-encoded, OpenCV convention).
cv::Mat LinearRgbToBgr(const cv::Mat& linear_rgb_f);

/// 32FC3 Lab (project D65) → 8UC3 BGR. Round-trip for diagnostics & previews.
cv::Mat LabToBgr(const cv::Mat& lab_f);

} // namespace ChromaPrint3D
