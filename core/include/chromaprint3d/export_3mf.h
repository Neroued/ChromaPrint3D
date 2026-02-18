#pragma once

/// \file export_3mf.h
/// \brief 3MF file export from ModelIR or pre-built meshes.

#include "mesh.h"
#include "color_db.h"

#include <cstdint>
#include <string>
#include <vector>

namespace ChromaPrint3D {

struct ModelIR;

/// Export a ModelIR to a 3MF file using default mesh config.
void Export3mf(const std::string& path, const ModelIR& model_ir);

/// Export a ModelIR to a 3MF file with custom mesh config.
void Export3mf(const std::string& path, const ModelIR& model_ir, const BuildMeshConfig& cfg);

/// Export a ModelIR to an in-memory 3MF buffer.
std::vector<uint8_t> Export3mfToBuffer(const ModelIR& model_ir,
                                       const BuildMeshConfig& cfg = {});

/// Export pre-built meshes to an in-memory 3MF buffer with given palette for object naming.
/// Mesh order must match: channel 0, channel 1, ..., channel N-1, [base].
std::vector<uint8_t> Export3mfFromMeshes(const std::vector<Mesh>& meshes,
                                          const std::vector<Channel>& palette,
                                          int base_channel_idx = -1,
                                          int base_layers = 0);

} // namespace ChromaPrint3D
