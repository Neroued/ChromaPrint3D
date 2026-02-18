#include "chromaprint3d/common.h"
#include "chromaprint3d/error.h"

namespace ChromaPrint3D {

std::string ToLayerOrderString(LayerOrder order) {
    switch (order) {
    case LayerOrder::Top2Bottom:
        return "Top2Bottom";
    case LayerOrder::Bottom2Top:
        return "Bottom2Top";
    }
    return "Top2Bottom";
}

LayerOrder FromLayerOrderString(const std::string& str) {
    if (str == "Top2Bottom") { return LayerOrder::Top2Bottom; }
    if (str == "Bottom2Top") { return LayerOrder::Bottom2Top; }
    throw FormatError("Invalid layer_order string: " + str);
}

} // namespace ChromaPrint3D
