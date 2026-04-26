#pragma once

/// \file color.h
/// \brief Umbrella header for the project's color types & conversions.
///
/// After the color-unification refactor, this header is a thin aggregator
/// over the `chromaprint3d/color/` sub-headers:
///
///   - `color/types.h`       : `Rgb`, `Lab`, `SrgbU8`, `Hsv` first-class types.
///   - `color/conversions.h` : sRGB↔linear, RGB↔XYZ↔Lab (project D65), HSV↔RGB.
///   - `color/distance.h`    : `DeltaE76`, `RgbDistanceSq`, `HsvDistanceBbs`,
///                             `LabHueAngleDeg`.
///   - `color/parse.h`       : `SrgbU8::FromHex/ToHex`, named colors,
///                             `ResolveColorLiteral`.
///   - `color/image.h`       : `cv::Mat`-level `BgrToLab` (self-implemented,
///                             never calls OpenCV `cvtColor(BGR2Lab)`), plus
///                             linear-RGB / preview entries.
///
/// **Lab math contract**: project-D65 (`(6/29)^3` threshold, true cbrt). This
/// is **not** byte-equivalent to OpenCV `cv::cvtColor(BGR2Lab)` (sub-ΔE drift
/// for historical ColorDB entries; see docs/development.md). The runtime code
/// path and all newly-generated data (modelpacks, calibration outputs) use
/// project-D65; historical `data/dbs/**/*.json entries[].lab` may still come
/// from OpenCV Lab and is **not** rebuilt in this refactor.

#include "chromaprint3d/color/types.h"
#include "chromaprint3d/color/conversions.h"
#include "chromaprint3d/color/distance.h"
#include "chromaprint3d/color/parse.h"
// `color/image.h` is intentionally *not* included here — it pulls in OpenCV.
// Modules that need `BgrToLab` etc. must include `color/image.h` directly.
