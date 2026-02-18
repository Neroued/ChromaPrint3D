/// \file detail/json_utils.h
/// \brief Internal JSON-related utility functions shared across core modules.

#pragma once

#include "chromaprint3d/common.h"
#include "chromaprint3d/error.h"

#include <nlohmann/json.hpp>

namespace ChromaPrint3D::detail {

/// Parse LayerOrder from a JSON value (accepts string or integer).
inline LayerOrder ParseLayerOrder(const nlohmann::json& value) {
    if (value.is_string()) { return FromLayerOrderString(value.get<std::string>()); }
    if (value.is_number_integer()) {
        int v = value.get<int>();
        if (v == 0) { return LayerOrder::Top2Bottom; }
        if (v == 1) { return LayerOrder::Bottom2Top; }
    }
    throw FormatError("Invalid layer_order value");
}

/// Parse LayerOrder from a JSON value with a fallback for null values.
inline LayerOrder ParseLayerOrder(const nlohmann::json& value, LayerOrder fallback) {
    if (value.is_null()) { return fallback; }
    return ParseLayerOrder(value);
}

} // namespace ChromaPrint3D::detail
