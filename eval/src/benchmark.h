#pragma once

#include "chromaprint3d/vectorize_eval.h"

namespace ChromaPrint3D::eval {

std::string MakeRunId();
std::string GitShortHash();
std::string CurrentTimestamp();

} // namespace ChromaPrint3D::eval
