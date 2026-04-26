# ChromaPrint3D Core

> [中文版](README.md)

ChromaPrint3D Core is a C++20 static library that provides a complete conversion pipeline from images to multi-color 3D printing models. It maps input image colors to multi-channel printing recipes, then generates slicer-ready 3MF model files through voxelization and meshing.

## Key Capabilities

- Image preprocessing (resize, denoise, alpha mask, color space conversion)
- ColorDB-based color matching and recipe generation (Lab / RGB color spaces)
- Optional ML model-assisted matching (ModelPackage)
- Recipe map (RecipeMap) to voxel grid (VoxelGrid) conversion
- Greedy meshing for triangle mesh generation
- 3MF model export (file or in-memory buffer)
- Calibration board generation and color database construction
- End-to-end conversion pipeline (`Convert` single-call API)

## Data Pipeline

### Image to 3MF Main Flow

```mermaid
flowchart LR
    A[Input Image] --> B[RasterProc]
    B --> C[RasterProcResult]
    C --> D["RecipeMap::MatchFromRaster"]
    D --> E[RecipeMap]
    E --> F["ModelIR::Build"]
    F --> G[VoxelGrids]
    G --> H["Mesh::Build"]
    H --> I["Export3mf"]
    I --> J[3MF File]
```

**Stage Details:**

| Stage | Input | Output | Description |
|-------|-------|--------|-------------|
| RasterProc | Image file/buffer | RasterProcResult | Resize, denoise, extract alpha mask, convert to linear RGB and Lab |
| MatchFromRaster | RasterProcResult + ColorDB | RecipeMap | SLIC/K-Means clustering (or per-pixel) followed by nearest-recipe matching |
| ModelIR::Build | RecipeMap + ColorDB | VoxelGrids | Expand per-pixel recipes into per-channel voxel occupancy grids |
| Mesh::Build | VoxelGrid | Mesh | Greedy meshing on each channel's voxels to produce triangle mesh |
| Export3mf | Mesh[] | 3MF | Write all channel meshes into 3MF format (built-in writer) |

### Calibration Board Flow

```mermaid
flowchart LR
    A[CalibrationBoardConfig] --> B[BuildCalibrationBoardMeta]
    B --> C[CalibrationBoardMeta]
    C --> D[BuildBoardModel]
    D --> E["ModelIR + Mesh"]
    E --> F[3MF Calibration Board]

    G[Board Photo + Meta] --> H[GenColorDBFromImage]
    H --> I[ColorDB]
```

Users first generate and print a calibration board 3MF, then photograph the printed result and combine it with the Meta to build a ColorDB, which can then be used in the image-to-3MF main flow.

## Directory Structure

```
core/
├── include/chromaprint3d/      # Public headers (available after installation)
│   ├── chromaprint3d.h         # Umbrella header
│   ├── version.h.in            # Version template (CMake generates version.h)
│   ├── export.h                # Export/visibility macros
│   ├── error.h                 # Exception hierarchy
│   ├── common.h                # Common enumerations
│   ├── vec3.h                  # 3D vector types
│   ├── color.h                 # Color types and color space conversions
│   ├── kdtree.h                # KD-Tree template
│   ├── color_db.h              # Color database
│   ├── imgproc.h               # Image preprocessing
│   ├── recipe_map.h            # Recipe mapping
│   ├── print_profile.h         # Print profile configuration
│   ├── model_package.h         # ML model package
│   ├── voxel.h                 # Voxel grids and ModelIR
│   ├── mesh.h                  # Triangle mesh
│   ├── export_3mf.h            # 3MF export
│   ├── calib.h                 # Calibration board
│   ├── pipeline.h              # Conversion pipeline
│   ├── encoding.h              # Image encoding
│   └── logging.h               # Logging initialization
│
├── src/                        # Source implementation (organized by module)
│   ├── common/                 # Common utilities
│   ├── color_db/               # ColorDB implementation
│   ├── imgproc/                # Image preprocessing
│   ├── match/                  # Color matching engine
│   │   └── detail/             # Internal helper headers
│   ├── calib/                  # Calibration board generation and detection
│   ├── geo/                    # Voxels, meshes, 3MF export
│   ├── pipeline/               # Conversion pipeline
│   ├── encoding/               # Image encoding
│   ├── logging/                # Logging initialization
│   └── detail/                 # Internal shared utilities
│       ├── cv_utils.h          # OpenCV helper functions
│       └── json_utils.h        # JSON parsing helpers
│
├── tests/                      # Unit tests
└── CMakeLists.txt              # Build configuration
```

## Module Reference

### common — Common Enumerations and Types

**Header:** `common.h`

Defines widely-used enumeration types throughout the library:

| Type | Description |
|------|-------------|
| `ResizeMethod` | Image resize algorithm (Nearest / Area / Linear / Cubic) |
| `DenoiseMethod` | Denoising algorithm (None / Bilateral / Median) |
| `LayerOrder` | Layer direction (Top2Bottom / Bottom2Top) |
| `ColorSpace` | Matching color space (Lab / Rgb) |

Provides `LayerOrder` string conversion functions: `ToLayerOrderString()` / `FromLayerOrderString()`.

### error — Exception Hierarchy

**Header:** `error.h`

All public functions throw subclasses of `ChromaPrint3D::Error` instead of bare `std::runtime_error`, enabling category-based catching:

| Exception | ErrorCode | Meaning |
|-----------|-----------|---------|
| `InputError` | InvalidInput | Invalid input arguments or data |
| `IOError` | IOError | File/stream I/O failure |
| `FormatError` | FormatError | Data format/parsing error (JSON, 3MF, etc.) |
| `ConfigError` | ConfigMismatch | Incompatible configuration (channel count, layer height, etc.) |
| `MatchError` | NoValidCandidate | Color matching found no usable candidate |

### vec3 / color — Math and Color Foundations

**Headers:** `vec3.h`, `color.h` (umbrella) + `color/{types,conversions,distance,parse,image}.h`

- `Vec3<T>`: Integer 3-component vector template, with `Vec3i = Vec3<int32_t>` (voxel coordinates, signed offsets) and `Vec3u = Vec3<uint32_t>` (mesh triangle indices, layout-compatible with `neroued_3mf::IndexTriangle` for zero-copy 3MF export)
- `Vec3f`: Float 3-component vector (normalization, distance, interpolation, clamping)

The `color/` submodule provides **four first-class types**:

- `Rgb`: Linear sRGB color (float [0, 1]), strong type (does **not** inherit from `Vec3f`); Lab/Hex round-trips
- `Lab`: CIE L\*a\*b\*, **project D65 Lab** (`(6/29)^3` threshold, D65 white point, true cbrt); `DeltaE76` provided
- `SrgbU8`: Gamma-encoded sRGB byte triple (no alpha); `FromHex` / `ToHex` is the **only** hex parser/formatter (strictly rejects 8-digit RGBA input)
- `Hsv`: HSV color, two semantic entries `FromRgb` (linear, scientific convention) and `FromSrgbU8` (gamma-encoded, BambuStudio byte-equivalent)

Color space conversion chain: sRGB gamma ↔ linear RGB ↔ XYZ (D65) ↔ L\*a\*b\*

**Lab math contract**: project D65 Lab is the **only** authoritative Lab implementation in the repository. `color/image.h::BgrToLab` self-implements the full BGR→linear→XYZ(D65)→Lab(D65) pipeline; calls to `cv::cvtColor(BGR2Lab)` are **forbidden**. The frontend `web/frontend/src/utils/colorConvert.ts` and Python `modeling/core/color_space.py::linear_rgb_to_lab_d65` share the same math.

Historical ColorDB entries (`data/dbs/**/*.json` `entries[*].lab`) come from OpenCV `cvtColor(BGR2Lab)` quantization and exhibit sub-ΔE76 drift versus project D65 (mean ~0.15 / p99 ~0.30 / max ~0.7). The project accepts this drift: historical ColorDB is **not** rebuilt, **no** `lab_math` metadata is introduced, and **no** modelpack-load compatibility check is performed; newly generated or progressively rebuilt ColorDBs naturally use project D65. See `docs/development.md` "Lab math" section.

### kdtree — KD-Tree Nearest Neighbor Search

**Header:** `kdtree.h`

Templated KD-Tree implementation supporting:
- `Nearest()` — Nearest neighbor query
- `KNearest()` — K nearest neighbor query
- `RadiusSearch()` — Radius-based range search

Uses projection functions for non-intrusive indexing; used by ColorDB for fast color lookups in Lab/RGB space.

### color_db — Color Database

**Header:** `color_db.h`

`ColorDB` is one of the core data structures, storing color recipes and their corresponding Lab values:

- `Entry`: A color record (Lab value + recipe vector)
- `Channel`: Channel descriptor (color name, material name)
- `ColorDB`: Complete database
  - JSON serialization/deserialization (`LoadFromJson` / `SaveToJson` / `FromJsonString` / `ToJsonString`)
  - KD-Tree indexed nearest neighbor queries (`NearestEntry` / `NearestEntries`)
  - Configuration: palette, layer height, line width, layer order

### imgproc — Image Preprocessing

**Header:** `imgproc.h`

The `RasterProc` class processes input images into the format required by the matching engine:

1. **Loading**: Supports file path, cv::Mat, or in-memory buffer
2. **Resizing**: Auto-adjusts dimensions based on scale / max_width / max_height
3. **Denoising**: Optional Bilateral / Median filtering
4. **Alpha mask extraction**: Generates pixel validity mask from alpha channel
5. **Color space conversion**: Outputs linear RGB (CV_32FC3) and CIE Lab (CV_32FC3)

Produces `RasterProcResult` containing `rgb`, `lab`, and `mask` matrices.

### match — Color Matching Engine

**Headers:** `recipe_map.h`, `print_profile.h`, `model_package.h`

Color matching is the library's core algorithm, mapping image pixels to the nearest printing recipe:

**Matching Flow:**
1. In non-dither mode, choose by `cluster_method`: `slic` uses superpixels, `kmeans` uses color clustering; target count `<=1` falls back to direct per-pixel matching
2. Query ColorDB with `k_candidates` nearest candidates and map recipes to the target PrintProfile
3. Optionally apply ModelPackage fallback gating (threshold/margin)
4. Write back `RecipeMap` (per-pixel recipe, mapped color, and source mask)

The non-dither main loop, cluster sample preparation, and cluster label writeback are OpenMP-parallelized.

**Key Types:**

| Type | Description |
|------|-------------|
| `RecipeMap` | Matching result: per-pixel recipes + mapped colors + source mask |
| `MatchConfig` | Matching parameters (candidate count, color space, clustering method, SLIC/K-Means knobs) |
| `MatchStats` | Matching statistics (total clusters, DB hits, model fallbacks, etc.) |
| `PrintProfile` | Print configuration (layer height, color layers, palette) |
| `ModelPackage` | ML model package with precomputed candidate recipes and predicted Lab values |
| `ModelGateConfig` | Model gate configuration (threshold, margin, enable flags) |

### geo — Geometry Processing

**Headers:** `voxel.h`, `mesh.h`

Converts recipe maps into 3D geometry:

- `VoxelGrid`: Dense boolean occupancy grid (H x W x L), one per channel
- `ModelIR`: Intermediate representation containing multiple VoxelGrids + palette info
- `ModelIR::Build()`: Builds voxel grids from RecipeMap + ColorDB (OpenMP parallelized)
- `Mesh`: Indexed triangle mesh (vertices + indices)
- `Mesh::Build()`: Greedy meshing to generate surface mesh from VoxelGrid

### export_3mf — 3MF Export

**Header:** `export_3mf.h`

Multi-channel 3MF model export based on the built-in 3MF writer (OPC ZIP + 3D model XML):

| Function | Description |
|----------|-------------|
| `Export3mf()` | Export ModelIR to a 3MF file |
| `Export3mfToBuffer()` | Export ModelIR to an in-memory buffer |
| `Export3mfFromMeshes()` | Export from pre-built Mesh vector (for caching scenarios) |

The writer emits standard 3MF by default. When a valid `SlicerPreset` (`SlicerPreset::machine_resolved()`
is `true`) is provided, it additionally injects private slicer parameter files under `Metadata/*`
(including BambuStudio cross-machine compatibility groups); if preset resolution fails, it safely falls
back to standard 3MF. `SlicerPreset` is constructed from `BambuPresetCatalog::Resolve(machine, nozzle,
layer_height)` using `data/presets/machines.json` + `data/preset_bases/<slug>_<lh>_<nozzle>.json`, with
runtime overrides from `data/presets/chromaprint_patches.json`.
ZIP packaging defaults to threshold-based `Store/Deflate`: small parts use Store, large parts prefer Deflate; if zlib is unavailable or compression fails, it safely falls back to Store.

Mesh building (voxel meshing, vector extrusion, and pre-export preprocessing) supports OpenMP
channel-level parallelism.

**Roadmap (V2)**
- Extend private slicer fields and vendor-specific metadata capabilities further via writer extension points, without breaking the standard 3MF path.

### calib — Calibration Board

**Header:** `calib.h`

The calibration board system establishes the mapping between actual printed colors and recipes:

**Board Generation:**
- `CalibrationBoardConfig`: Board configuration (channel count, color layers, layout params)
- `CalibrationRecipeSpec`: Recipe specification (`num_channels^color_layers` recipe permutations)
- `GenCalibrationBoard()`: Generate calibration board 3MF + Meta JSON
- `GenCalibrationBoardMeshes()`: Generate intermediate mesh results (for server-side geometry caching)

**ColorDB Construction:**
- `GenColorDBFromImage()`: Build ColorDB from calibration board photo + Meta
- Supports both file path and in-memory buffer inputs

**Caching Optimization:**
- `CalibrationBoardMeshes`: Stores pre-built mesh vectors
- `BuildResultFromMeshes()`: Quickly rebuild results from cached meshes + new palette

### pipeline — Conversion Pipeline

**Header:** `pipeline.h`

The `Convert()` function provides an end-to-end image-to-3MF pipeline:

```cpp
ConvertResult result = Convert(request);
```

`ConvertRasterRequest` encapsulates all parameters:
- Image input (path or buffer)
- ColorDB input (paths or preloaded instances)
- Optional ModelPackage
- Image processing parameters (scale, max_width/height)
- Matching parameters (print_mode, color_space, k_candidates, cluster_method, cluster_count, slic_*)
- Geometry parameters (flip_y, pixel_mm, layer_height_mm)
- Output control (whether to generate preview and source mask images)

`ConvertResult` contains:
- `model_3mf`: 3MF model buffer
- `preview_png`: Preview image PNG buffer
- `source_mask_png`: Source mask PNG buffer
- `stats`: Matching statistics

Supports `ProgressCallback` for stage-based progress reporting (LoadingResources → Preprocessing → Matching → BuildingModel → Exporting).

### encoding — Image Encoding

**Header:** `encoding.h`

Encodes OpenCV Mat to image formats:
- `EncodePng()`: Encode to PNG buffer
- `EncodeJpeg()`: Encode to JPEG buffer (configurable quality)
- `SaveImage()`: Save to file

### logging — Logging

**Header:** `logging.h`

spdlog-based logging initialization:
- `InitLogging()`: Set up global logger and level
- `ParseLogLevel()`: Parse log level from string

## External Dependencies

| Dependency | Purpose | Linking |
|------------|---------|---------|
| **OpenCV** | Image loading, resizing, denoising, color conversion, encoding | `opencv_core` `opencv_imgproc` `opencv_imgcodecs` |
| **Built-in 3MF Writer** | 3MF export (OPC/ZIP + XML) | Implemented inside core |
| **zlib** | 3MF ZIP Deflate backend (optional) | `ZLIB::ZLIB` (enabled when found) |
| **spdlog** | Structured logging | `spdlog::spdlog_header_only` |
| **nlohmann/json** | JSON serialization (header-only, via 3dparty directory) | Header-only |
| **OpenMP** | Optional parallel acceleration (matching, voxel/mesh building, calibration statistics, layer previews) | `OpenMP::OpenMP_CXX` (auto-detected) |

## Building and Integration

### CMake Target

```cmake
# Library target
add_library(chromaprint3d STATIC)
add_library(ChromaPrint3D::core ALIAS chromaprint3d)
```

### Using in a Parent Project

```cmake
target_link_libraries(your_target PRIVATE ChromaPrint3D::core)
```

Linking `ChromaPrint3D::core` automatically propagates all public dependencies (OpenCV, spdlog, OpenMP).

Parallel tuning tip: if your host service uses a task thread pool, tune `OMP_NUM_THREADS` together with
task concurrency to avoid oversubscription from `task_concurrency × OpenMP_threads`.

### Version Numbers

Version numbers are managed centrally via `PROJECT_VERSION` in the top-level `CMakeLists.txt`, generated through `configure_file()` into `version.h`:

```cpp
#include <chromaprint3d/version.h>

CHROMAPRINT3D_VERSION_MAJOR   // Major version
CHROMAPRINT3D_VERSION_MINOR   // Minor version
CHROMAPRINT3D_VERSION_PATCH   // Patch version
CHROMAPRINT3D_VERSION_STRING  // Full version string, e.g. "1.2.0"
```

### Umbrella Header

```cpp
#include <chromaprint3d/chromaprint3d.h>  // Includes all public headers
```

Or include individual headers as needed:

```cpp
#include <chromaprint3d/pipeline.h>   // Conversion pipeline only
#include <chromaprint3d/color_db.h>   // Color database only
```
