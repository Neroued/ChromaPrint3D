#pragma once

#include <cstdint>

namespace ChromaPrint3D {

enum class ResizeMethod : uint8_t {
    Nearest = 0,
    Area    = 1,
    Linear  = 2,
    Cubic   = 3,
};

enum class DenoiseMethod : uint8_t {
    None      = 0,
    Bilateral = 1,
    Median    = 2,
};

enum class LayerOrder : uint8_t {
    Top2Bottom = 0,
    Bottom2Top = 1,
};

enum class ColorSpace : uint8_t {
    Lab = 0,
    Rgb = 1,
};


} // namespace ChromaPrint3D