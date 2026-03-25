# Core 模块索引

## 模块职责

`core/` 是 C++20 核心库，负责图像预处理、颜色匹配、体素/网格构建、3MF 导出，以及校准板和 ColorDB 相关能力。

## 关键目录与入口

- 公共 API：`core/include/chromaprint3d/`
- 核心实现：`core/src/`
  - 图像处理：`imgproc/`
  - 匹配算法：`match/`
  - 几何与导出：`geo/`、`vecgeo/`
  - 转换编排：`pipeline/`
  - 校准：`calib/`

## 常见改动落点

| 目标 | 入口文件 |
|---|---|
| 修改栅格预处理 | `core/src/imgproc/raster_proc.cpp` |
| 修改颜色匹配主流程 | `core/src/match/match.cpp` |
| 修改候选筛选/评分 | `core/src/match/candidate_select.cpp` |
| 修改配方与打印配置映射 | `core/src/match/recipe_map.cpp`、`core/src/match/print_profile.cpp` |
| 修改配方区域编辑 | `core/src/match/raster_region_map.cpp`、`core/src/match/recipe_alternatives.cpp` |
| 修改体素/网格生成 | `core/src/geo/geo.cpp`、`core/src/vecgeo/vector_mesh_builder.cpp` |
| 修改 3MF 导出 | `core/src/geo/export_3mf.cpp` |
| 修改 Bambu 预设与耗材元数据 | `core/src/geo/bambu_metadata.cpp` |
| 修改 SVG 几何提取与遮挡裁剪 | `core/src/imgproc/vector_proc.cpp`、`core/src/imgproc/occlusion.cpp`、`core/src/imgproc/gradient.cpp` |
| 修改端到端转换编排 | `core/src/pipeline/pipeline.cpp`、`core/src/pipeline/vector_pipeline.cpp` |
| 修改校准板与 ColorDB 构建 | `core/src/calib/calib.cpp` |

## Bambu Studio 预设体系

`core/include/chromaprint3d/common.h` 定义了 `NozzleSize`（N02/N04）和 `FaceOrientation`（FaceUp/FaceDown）枚举。`PrintProfile` 携带这两个字段，通过 `SlicerPreset::FromProfile` 选择 4 种预设 JSON 之一（`data/presets/bambu_p2s_0.08mm_{n02|n04}_{faceup|facedown}.json`）。同时，`face_orientation` 也会联动 3MF 导出几何：`FaceDown` 时模型整体绕 Y 轴旋转 180°，`FaceUp` 保持原方向。

颜色匹配逻辑在 `bambu_metadata.cpp` 中：`MatchColorToSlot` 基于 RGB 欧氏距离将每个 mesh 颜色匹配到预设 `filament_colour` 数组中最近的槽位。预设中的耗材丝信息（颜色、类型、温度等）完整保留到 3MF 输出中。导出翻转逻辑位于 `export_3mf.cpp`，与 `flip_y`（坐标系适配）解耦。

## 输入输出契约（高频）

- 栅格转换编排：`ConvertRasterRequest` -> `ConvertResult`
- 栅格匹配阶段：`ConvertRasterRequest` -> `MatchRasterResult`（两阶段）
- 栅格模型生成：`MatchRasterResult` -> `ConvertResult`（两阶段）
- 矢量转换编排：`ConvertVectorRequest` -> `ConvertResult`
- 匹配结果：`RecipeMap`
- 区域映射：`RasterRegionMap`（`raster_region_map.h`）
- 候选配方：`RecipeCandidate`（`recipe_alternatives.h`）
- 中间几何表示：`ModelIR` / `VoxelGrid`

如需改公共接口，优先从 `core/include/chromaprint3d/*.h` 对照实现文件同步修改。

## 变更注意事项

- 新增工具函数前，先检索现有实现，避免重复语义函数。
- 能局部使用的工具函数放匿名命名空间；多模块复用再升级为公共函数。
- `.cpp` 内优先通过定义顺序解决依赖，避免无必要前向声明。
- 修改 C++ 后仅对本次变更文件执行 `clang-format -i`。

## 最小验证

```bash
cmake --build build -j$(nproc)
```

如改动覆盖转换行为，建议再执行一次端到端 CLI 冒烟（示例）：

```bash
build/bin/raster_to_3mf --image input.png --db data/dbs/<your_db>.json --out /tmp/out.3mf
```

## 相关任务手册

- [docs/agents/tasks/tune_color_match.md](../tasks/tune_color_match.md)
- [docs/agents/tasks/extend_cli_flag.md](../tasks/extend_cli_flag.md)

