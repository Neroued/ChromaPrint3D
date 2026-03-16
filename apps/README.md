# ChromaPrint3D Apps 指南

## 构建与运行位置

- 构建步骤请看：[docs/build.md](../docs/build.md)
- 本地开发与联调请看：[docs/development.md](../docs/development.md)
- 构建完成后可执行文件位于：`build/bin/`

## 可执行工具清单

当前 `apps/CMakeLists.txt` 定义了 7 个 CLI：

| 可执行文件 | 用途 |
|---|---|
| `gen_calibration_board` | 生成校准板 3MF 与元数据 |
| `build_colordb` | 从校准板照片构建 ColorDB |
| `raster_to_3mf` | 栅格图像转多色 3MF |
| `svg_to_3mf` | SVG 转多色 3MF |
| `gen_representative_board` | 从配方集生成代表性校准板 |
| `gen_stage` | 生成阶梯校准模型 |
| `gen_test_preset_3mf` | 生成预设映射测试 3MF（调试工具） |

> 矢量化 CLI 工具（raster_to_svg、evaluate_svg）已迁移至独立库 [neroued_vectorizer](https://github.com/neroued/neroued_vectorizer)。

## 典型使用流程

### 流程 A：校准并转换栅格图

```mermaid
flowchart TD
    genBoard[gen_calibration_board] -->|board.3mf + board.json| printBoard[打印并拍摄校准板]
    printBoard --> buildDb[build_colordb]
    buildDb -->|color_db.json| rasterConvert[raster_to_3mf]
    inputImage[输入图像] --> rasterConvert
    rasterConvert -->|output.3mf| slicer[切片打印]
```

### 流程 B：矢量链路

```mermaid
flowchart TD
    svgInput[输入 SVG] --> svg2mf[svg_to_3mf]
    svg2mf --> modelOut[输出 3MF]
```

> 栅格图转 SVG（raster_to_svg）已迁移至 [neroued_vectorizer](https://github.com/neroued/neroued_vectorizer)。

## 各工具快速参考

下列命令均假设在仓库根目录执行，并使用 `build/bin/` 前缀。

### 1) `gen_calibration_board`

用途：生成校准板模型与元数据。  
示例：

```bash
./build/bin/gen_calibration_board --channels 4 --out board.3mf --meta board.json
```

### 2) `build_colordb`

用途：根据校准板照片 + 元数据生成 ColorDB。  
示例：

```bash
./build/bin/build_colordb --image calib_photo.png --meta board.json --out color_db.json
```

### 3) `raster_to_3mf`

用途：栅格图像转换为多色 3MF。  
示例：

```bash
./build/bin/raster_to_3mf --image input.png --db color_db.json --out output.3mf

# 显式覆盖底板层数并开启双面生成
./build/bin/raster_to_3mf --image input.png --db color_db.json --out output.3mf \
  --base-layers 6 --double-sided 1
```

### 4) `svg_to_3mf`

用途：SVG 直接转换为多色 3MF。  
示例：

```bash
./build/bin/svg_to_3mf --svg input.svg --db color_db.json --out output.3mf

# 显式覆盖底板层数并开启双面生成
./build/bin/svg_to_3mf --svg input.svg --db color_db.json --out output.3mf \
  --base-layers 6 --double-sided 1
```

### 5) `gen_representative_board`

用途：从配方集生成代表性校准板。  
示例：

```bash
./build/bin/gen_representative_board --recipes recipes.json --ref-db color_db.json --out board.3mf
```

### 6) `gen_stage`

用途：生成固定阶梯校准模型。  
示例：

```bash
./build/bin/gen_stage --out stage.3mf
```

### 7) `gen_test_preset_3mf`

用途：生成切片预设/AMS 映射测试用 3MF（开发调试用途）。  
示例（位置参数）：

```bash
./build/bin/gen_test_preset_3mf test_8color_preset.3mf data/presets
```

## 参数与行为说明

- `apps` 下工具的完整参数以各工具 `--help` 输出为准。
- `gen_test_preset_3mf` 当前使用位置参数，不提供 `--help` 选项。
- `raster_to_3mf` / `svg_to_3mf` 支持：
  - `--base-layers N`：覆盖底板层数（`-1` 表示继承 ColorDB 默认值，`0` 表示无底板）
  - `--double-sided 0|1`：是否生成双面镜像色层结构
