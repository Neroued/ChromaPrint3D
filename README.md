# ChromaPrint3D

ChromaPrint3D 是一个将图像颜色映射为多通道配方并导出 3MF 的 C++20 核心库，提供命令行工具与 Web 服务，覆盖校准板生成、ColorDB 构建与图像转 3MF 流程。

## 功能
- ColorDB 加载/保存与最近邻匹配（Lab/RGB）
- 图像预处理（缩放、去噪、透明度 mask）
- 配方匹配与模型中间表示（RecipeMap/ModelIR）
- 体素到网格并导出 3MF（lib3mf）
- 校准板与颜色数据库生成

## 依赖
- CMake >= 3.25
- C++20 编译器
- 子模块：OpenCV、lib3mf、spdlog

## 获取代码
```
git submodule update --init --recursive
```

## 构建
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

生成的可执行文件在 `build/bin/` 下：
- `gen_calibration_board`
- `build_colordb`
- `image_to_3mf`
- `chromaprint3d_server`

## 命令行示例
1) 生成校准板与元数据
```
build/bin/gen_calibration_board --channels 4 --out calibration_board.3mf --meta calibration_board.json
```
2) 从拍摄的校准板图生成 ColorDB
```
build/bin/build_colordb --image calib.png --meta calibration_board.json --out color_db.json
```
3) 从图像生成 3MF（并输出预览图）
```
build/bin/image_to_3mf --image input.png --db color_db.json --out output.3mf --preview output_preview.png
```

## 目录结构
- `core/`：C++ 核心实现
- `apps/`：命令行工具与 HTTP 服务器
- `web/`：Web 前端
- `3dparty/`：第三方依赖子模块
