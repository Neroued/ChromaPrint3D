#pragma once

/// \file export_3mf.h
/// \brief 3MF file export from ModelIR or pre-built meshes.

#include "mesh.h"
#include "color_db.h"
#include "slicer_preset.h"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ChromaPrint3D {

struct ModelIR;

/// Export a ModelIR to a 3MF file using default mesh config.
void Export3mf(const std::string& path, const ModelIR& model_ir);

/// Export a ModelIR to a 3MF file with custom mesh config.
void Export3mf(const std::string& path, const ModelIR& model_ir, const BuildMeshConfig& cfg);

/// Export a ModelIR to an in-memory 3MF buffer.
/// When face_orientation is FaceDown, geometry is rotated 180 degrees around Y.
std::vector<uint8_t> Export3mfToBuffer(const ModelIR& model_ir, const BuildMeshConfig& cfg = {},
                                       FaceOrientation face_orientation = FaceOrientation::FaceUp);

/// Export a ModelIR to an in-memory 3MF buffer with preset-compatible signature.
/// When preset_json_path is available, private slicer metadata attachments are injected;
/// otherwise the export safely falls back to standard 3MF.
std::vector<uint8_t> Export3mfToBuffer(const ModelIR& model_ir, const BuildMeshConfig& cfg,
                                       const SlicerPreset& preset,
                                       FaceOrientation face_orientation = FaceOrientation::FaceUp);

/// Export pre-built meshes to an in-memory 3MF buffer with given palette for object naming.
/// Mesh order must match: channel 0, channel 1, ..., channel N-1, [base].
std::vector<uint8_t>
Export3mfFromMeshes(const std::vector<Mesh>& meshes, const std::vector<Channel>& palette,
                    int base_channel_idx = -1, int base_layers = 0,
                    FaceOrientation face_orientation = FaceOrientation::FaceUp);

/// Export pre-built meshes with preset-compatible signature.
/// When preset_json_path is available, private slicer metadata attachments are injected;
/// otherwise the export safely falls back to standard 3MF.
std::vector<uint8_t> Export3mfFromMeshes(const std::vector<Mesh>& meshes,
                                         const std::vector<Channel>& palette, int base_channel_idx,
                                         int base_layers, const SlicerPreset& preset,
                                         FaceOrientation face_orientation = FaceOrientation::FaceUp,
                                         const std::string& model_name    = {});

/// Export pre-built meshes with explicit per-mesh names (no slicer preset).
std::vector<uint8_t>
Export3mfFromMeshes(const std::vector<Mesh>& meshes, const std::vector<Channel>& palette,
                    const std::vector<std::string>& names,
                    FaceOrientation face_orientation = FaceOrientation::FaceUp);

/// Export pre-built meshes with explicit per-mesh name and slot assignment.
/// \param palette Slot-ordered palette (slot 1..N) used to map display colors.
/// \param names Per-mesh display name.
/// \param slots Per-mesh 1-based slot assignment.
std::vector<uint8_t> Export3mfFromMeshes(const std::vector<Mesh>& meshes,
                                         const std::vector<Channel>& palette,
                                         const std::vector<std::string>& names,
                                         const std::vector<int>& slots, const SlicerPreset& preset,
                                         FaceOrientation face_orientation = FaceOrientation::FaceUp,
                                         const std::string& model_name    = {});

// ── Direct-to-file variants (no in-memory buffer) ────────────────────────────

/// Export a ModelIR directly to a 3MF file with face orientation.
void Export3mfToFile(const std::string& path, const ModelIR& model_ir, const BuildMeshConfig& cfg,
                     FaceOrientation face_orientation);

/// Export a ModelIR directly to a 3MF file with preset and face orientation.
void Export3mfToFile(const std::string& path, const ModelIR& model_ir, const BuildMeshConfig& cfg,
                     const SlicerPreset& preset, FaceOrientation face_orientation);

/// Export pre-built meshes directly to a 3MF file.
void Export3mfFromMeshesToFile(const std::string& path, const std::vector<Mesh>& meshes,
                               const std::vector<Channel>& palette, int base_channel_idx,
                               int base_layers, FaceOrientation face_orientation);

/// Export pre-built meshes directly to a 3MF file with preset.
void Export3mfFromMeshesToFile(const std::string& path, const std::vector<Mesh>& meshes,
                               const std::vector<Channel>& palette, int base_channel_idx,
                               int base_layers, const SlicerPreset& preset,
                               FaceOrientation face_orientation,
                               const std::string& model_name = {});

// ── Mesh building utilities (exposed for pipeline use) ────────────────────────

/// Build per-channel + base meshes from a ModelIR using greedy meshing.
std::vector<Mesh> BuildMeshes(const ModelIR& model_ir, const BuildMeshConfig& cfg);

/// Build a transparent layer mesh covering the full XY occupancy of all voxel grids.
/// The slab spans [max_z, max_z + thickness] in the pre-flip coordinate space.
Mesh BuildTransparentLayerFromModelIR(const ModelIR& model_ir, float pixel_mm,
                                      float layer_height_mm, float thickness_mm);

/// Per-mesh name and 1-based slot assignment for explicit 3MF export.
struct MeshNamesAndSlots {
    std::vector<std::string> names;
    std::vector<int> slots;
};

/// Build names and slot assignment for a mesh list ordered:
///   [ch0..chN-1, base?, transparent?].
MeshNamesAndSlots BuildMeshNamesAndSlots(size_t mesh_count, const std::vector<Channel>& palette,
                                         int base_channel_idx, int base_layers,
                                         bool has_transparent_layer);

// ── File utility (used by pipeline for atomic writes) ─────────────────────────

namespace detail {

/// Atomically write a byte buffer to disk (write temp + rename).
void WriteBufferToFileAtomically(const std::filesystem::path& final_path,
                                 const std::vector<uint8_t>& bytes);

} // namespace detail

} // namespace ChromaPrint3D
