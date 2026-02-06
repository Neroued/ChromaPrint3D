# ChromaPrint3D

ChromaPrint3D 是一个将图像颜色映射为多通道配方并导出 3MF 的 C++20 核心库，提供 Python 绑定与命令行工具，覆盖校准板生成、ColorDB 构建与图像转 3MF 流程。

## 功能
- ColorDB 加载/保存与最近邻匹配（Lab/RGB）
- 图像预处理（缩放、去噪、透明度 mask）
- 配方匹配与模型中间表示（RecipeMap/ModelIR）
- 体素到网格并导出 3MF（lib3mf）
- 校准板与颜色数据库生成

## 依赖
- CMake >= 3.25
- C++20 编译器
- Python >= 3.10（仅 Python 绑定）
- numpy, scikit-build-core
- 子模块：OpenCV、lib3mf、pybind11

## 获取代码
```
git submodule update --init --recursive
```

## 构建（C++ 库与工具）
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

生成的可执行文件在 `build/apps/` 下：
- `gen_calibration_board`
- `build_colordb`
- `image_to_3mf`

## 安装 Python 绑定
```
python -m pip install -U pip
python -m pip install .
```

## 命令行示例
1) 生成校准板与元数据
```
build/apps/gen_calibration_board --channels 4 --out calibration_board.3mf --meta calibration_board.json
```
2) 从拍摄的校准板图生成 ColorDB
```
build/apps/build_colordb --image calib.png --meta calibration_board.json --out color_db.json
```
3) 从图像生成 3MF（并输出预览图）
```
build/apps/image_to_3mf --image input.png --db color_db.json --out output.3mf --preview output_preview.png
```

## Python 示例
```
import ChromaPrint3D as c

db = c.ColorDB.load_from_json("color_db.json")
img = c.ImgProc().run("input.png")

cfg = c.MatchConfig()
cfg.k_candidates = 1
cfg.color_space = c.ColorSpace.Lab
recipe_map = c.RecipeMap.MatchFromImage(img, db, cfg)

model = c.ModelIR.Build(recipe_map, db, c.BuildModelIRConfig())
c.Export3mf("output.3mf", model, c.BuildMeshConfig())
```

完整脚本可参考 `scrips/test_import.py`（支持预览图输出，PNG 需要 Pillow 或 imageio）。

## 目录结构
- `core/`：C++ 核心实现
- `python/`：pybind11 绑定与类型声明
- `apps/`：命令行工具
- `3dparty/`：第三方依赖子模块