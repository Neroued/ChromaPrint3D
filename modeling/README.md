# ChromaPrint3D — 薄层叠色建模系统

## 目录

- [概述](#概述)
- [物理模型](#物理模型)
  - [基本假设](#基本假设)
  - [前向模型公式](#前向模型公式)
  - [参数说明](#参数说明)
  - [标定分阶段设计](#标定分阶段设计)
- [目录结构](#目录结构)
- [快速开始](#快速开始)
- [完整流水线](#完整流水线)
  - [Step 1 — 提取单色阶梯数据](#step-1--提取单色阶梯数据)
  - [Step 2 — 拟合 Stage A (E, k)](#step-2--拟合-stage-a-e-k)
  - [Step 3 — 拟合 Stage B (E, k, γ, δ, C₀)](#step-3--拟合-stage-b-e-k-γ-δ-c₀)
  - [Step 4 — 选取代表性配方](#step-4--选取代表性配方)
  - [Step 5 — 构建运行时模型包](#step-5--构建运行时模型包)
- [评估工具](#评估工具)
  - [预览图质量评估](#预览图质量评估)
  - [单色通道曲线绘制](#单色通道曲线绘制)
  - [Stage A 诊断图](#stage-a-诊断图)
- [核心模块 (core/)](#核心模块-core)
- [数据文件说明](#数据文件说明)
  - [ColorDB 格式](#colordb-格式)
  - [输出文件](#输出文件)
- [依赖项](#依赖项)

---

## 概述

本建模系统为 ChromaPrint3D 的核心 Python 子包，实现了一套**基于 Beer-Lambert 指数衰减模型的薄层叠色打印颜色预测系统**。

**核心能力：** 给定 8 种可选打印颜色（例如 CMYK + RYBW）的叠层顺序和层高，预测打印后的最终显示颜色。

**应用场景：**
1. 根据目标图像自动选择每个像素点的打印配方（颜色层序列）
2. 选取代表性配方集用于色板打印和迭代标定
3. 评估模型预测精度与预览渲染质量

系统通过五步流水线完成从原始拍照数据到运行时模型包的全流程：

```
拍照标注 → 单色参数拟合 → 多色联合拟合 → 配方选取 → 模型包构建 → C++ 运行时
```

---

## 物理模型

### 基本假设

1. **Beer-Lambert 吸收定律：** 每种颜色的薄层对光的吸收遵循指数衰减规律。光线从顶层进入、穿过各层后反射回来，最终颜色由各层的吸收和基板颜色共同决定。

2. **微层离散化：** 将实际打印层离散为更细的"微层"（默认 0.04mm/层）进行递推计算，以提高数值精度。例如 0.08mm 层高的一层颜色会被展开为 2 个微层。

3. **线性 RGB 空间：** 模型的中间计算在线性 RGB 空间中进行（而非 sRGB 的 gamma 空间），因为光的物理吸收过程在线性空间中是可叠加的。

4. **独立通道假设 (Stage A)：** 在初始标定阶段，假设三个 RGB 通道的吸收独立——即每种颜色的红、绿、蓝吸收系数分别拟合。

5. **相邻层交互 (Stage B)：** 在进阶标定中引入邻层交互项 δ[i, prev]，建模相邻颜色层之间可能存在的化学/光学耦合效应（如渗透、散射变化）。

6. **层高缩放 (Stage B)：** 不同层高下，材料的有效吸收系数可能不同（例如 0.08mm 层高的材料密度与 0.04mm 不同），通过缩放因子 γ 来修正。

7. **基板颜色可学习：** 基板（如白色 PLA）的初始颜色 C₀ 不完全是理论白色，可以从数据中学习得到更准确的值。

### 前向模型公式

模型从底板颜色 C₀ 开始，**自底向上**逐微层递推：

```
对于每个微层 i (从底到顶):
    scale_i = exp(γ[color_i] × h_delta)            # 层高缩放
    k_eff   = k[color_i] × scale_i + δ[color_i, prev_color]  # 有效吸收系数
    T_i     = exp(-k_eff × t)                       # 透射率
    C_next  = (1 - T_i) × E[color_i] + T_i × C_prev  # 颜色递推

其中:
    h_delta = layer_height / height_ref - 1.0       # 层高偏移量
    t       = micro_layer_height                    # 微层厚度 (mm)
```

**直觉理解：**
- 当 k_eff × t 很大（厚层或高吸收）：T → 0，C → E（平衡色，即该颜色无限厚时的外观）
- 当 k_eff × t 很小（薄层或低吸收）：T → 1，C → C_prev（几乎透明，看到下层颜色）

### 参数说明

| 参数 | 形状 | 含义 |
|------|------|------|
| **E** | (N_colors, 3) | 平衡色——每种颜色在无限厚时的 linear RGB 外观 |
| **k** | (N_colors, 3) | 吸收系数——每种颜色的 RGB 三通道吸收速率 |
| **γ (gamma)** | (N_colors,) | 层高缩放因子——修正不同层高下的 k 值 |
| **δ (delta)** | (N_colors, N_colors, 3) | 邻层交互——颜色 i 的下层是颜色 j 时的 k 修正值 |
| **C₀** | (N_substrates, 3) | 基板初始颜色 (linear RGB) |
| **t** | 标量 | 微层厚度，默认 0.04mm |
| **height_ref** | 标量 | 层高参考值 (mm)，用于计算 h_delta |

其中 N_colors = 8（当前配置），N_substrates ≥ 1。

### 标定分阶段设计

| 阶段 | 拟合参数 | 数据来源 | 损失空间 |
|------|----------|----------|----------|
| **Stage A** | E, k（仅单色） | 单色阶梯色板 (1_single_stage.json) | linear RGB |
| **Stage B** | E, k, γ, δ, C₀ | 多色组合实测 (ColorDB) | OpenCV Lab |

Stage A 提供初始估计，Stage B 在此基础上联合优化所有参数。Stage B 采用 Lab 空间损失是因为 Lab 更接近人眼感知的颜色差异。

---

## 目录结构

```
modeling/
├── __init__.py                 # 包入口
├── README.md                   # 本文档
│
├── core/                       # 核心可复用模块 (7个)
│   ├── color_space.py          # 色彩空间转换 (sRGB/linear/XYZ/Lab/OpenCV Lab)
│   ├── math_utils.py           # 数学工具 (sigmoid/softplus/logit 等)
│   ├── adam.py                 # 通用 Adam 优化器
│   ├── io_utils.py             # JSON 读取、标签标准化、路径解析
│   ├── forward_model.py        # 前向模型加载与批量预测
│   ├── colorchecker_tool.py    # ColorChecker 色卡工具类
│   └── color_calibration.py    # 图像颜色校准 (CCM 3×3)
│
├── pipeline/                   # 标定流水线 (5步)
│   ├── step1_extract_stages.py # 提取单色阶梯数据
│   ├── step2_fit_stage_a.py    # 拟合 Stage A
│   ├── step3_fit_stage_b.py    # 拟合 Stage B
│   ├── step4_select_recipes.py # 选取代表性配方
│   └── step5_build_model_package.py  # 构建运行时模型包
│
├── eval/                       # 分析评估工具 (3个)
│   ├── preview_quality.py      # 预览图质量评估 (BADE)
│   ├── plot_stage_channels.py  # 单色通道响应曲线
│   └── plot_stage_a_diagnostics.py  # Stage A 诊断图
│
├── dbs/                        # ColorDB 输入数据
│   ├── CMYK_008.json
│   ├── RYBW_008_corrected.json
│   └── ...
│
├── data -> /mnt/.../chroma     # 原始拍照数据 (symlink)
│
└── output/                     # 所有生成产物
    ├── params/                 # 模型参数 JSON
    ├── packages/               # 运行时模型包 (供 C++ 端使用)
    ├── recipes/                # 代表性配方集
    ├── reports/                # 评估报告
    └── previews/               # 3MF 文件与预览图
```

---

## 快速开始

所有脚本均从**项目根目录**以 `python -m` 方式运行：

```bash
cd /path/to/ChromaPrint3D

# 示例：运行 Stage A 拟合
python -m modeling.pipeline.step2_fit_stage_a

# 示例：运行 Stage B 拟合
python -m modeling.pipeline.step3_fit_stage_b \
    --db modeling/dbs \
    --stage-a modeling/output/params/stage_A_from_stairs.json

# 示例：评估预览质量
python -m modeling.eval.preview_quality \
    --images xhs1 columbina --modes colordb_only model_only
```

> **注意：** 运行目录必须是项目根目录（`ChromaPrint3D/`），因为包导入依赖 `modeling/` 作为顶层包。

---

## 完整流水线

### 数据流概览

```
   ┌─────────────────────────────────────────────────────────────────┐
   │                         标定数据准备                            │
   │  拍摄带 ColorChecker 的单色阶梯色板照片 → LabelMe 标注          │
   └───────────────────────────┬─────────────────────────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Step 1: 提取阶梯数据  │
                    │  → 1_single_stage.json │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Step 2: Stage A 拟合  │
                    │  → stage_A_*.json      │
                    └──────────┬──────────┘
                               │
            ┌──────────────────┤
            │                  ▼
    ColorDB (实测)    ┌─────────────────────┐
     dbs/*.json ────→ │  Step 3: Stage B 拟合  │
                      │  → stage_B_*.json      │
                      └──────────┬──────────┘
                                 │
                    ┌────────────┤────────────┐
                    ▼                         ▼
         ┌──────────────────┐      ┌──────────────────┐
         │ Step 4: 选取配方   │      │ Step 5: 模型包    │
         │ → recipes_*.json  │      │ → model_package_* │
         └────────┬─────────┘      └────────┬─────────┘
                  │                          │
                  ▼                          ▼
        gen_representative_board       image_to_3mf
              (C++ 端)                   (C++ 端)
```

---

### Step 1 — 提取单色阶梯数据

从带 ColorChecker 的标注照片中，提取每种颜色在不同层数下的实测颜色值。

```bash
python -m modeling.pipeline.step1_extract_stages \
    --input-root modeling/data/1_single \
    --output modeling/output/params/1_single_stage.json
```

**输入：** 目录结构为 `{input-root}/{substrate}/{color}/`，每个子目录含 LabelMe JSON 标注和对应图片。

**处理流程：**
1. 从标注中找到 ColorChecker 色卡的四角和阶梯色板区域
2. 利用 ColorChecker 参考值计算 3×3 颜色校正矩阵 (CCM)
3. 在色板的 6×3 网格中采样每个色块的颜色
4. 自动判断色板朝向（通过基板色块的 Lab 距离）
5. 对多张照片的结果取均值，输出标准化 JSON

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input-root` | `modeling/data/1_single` | 含 `{基板}/{颜色}/` 子目录的根目录 |
| `--output` | `modeling/output/params/1_single_stage.json` | 输出 JSON 路径 |
| `--ref-json` | `modeling/data/ColorChecker/ColorCheckerjson.json` | ColorChecker 参考 LabelMe JSON |
| `--method` | `mean` | 色块采样方法 (`mean` / `median`) |
| `--shrink` | `0.85` | 色块多边形缩小比例（避免边缘） |
| `--agg-method` | `mean` | 多图聚合方法 |
| `--workers` | `0` (自动) | 并行线程数 |

---

### Step 2 — 拟合 Stage A (E, k)

从单色阶梯数据拟合每种颜色的基本参数——平衡色 E 和吸收系数 k。

```bash
python -m modeling.pipeline.step2_fit_stage_a \
    --input modeling/output/params/1_single_stage.json \
    --output modeling/output/params/stage_A_parameters.json
```

**拟合原理：** 对于单色涂层，Beer-Lambert 模型简化为：

```
C(n) = E + (C₀ - E) × exp(-k × n × layer_height)
```

其中 n 是层数。通过 Adam 优化器最小化预测值与实测值在 linear RGB 空间的 MSE 损失，使用加权方案（低层数权重更高）。

参数空间采用重参数化以确保物理约束：
- E = sigmoid(u)，保证 E ∈ [0, 1]
- k = softplus(v)，保证 k > 0

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `output/params/1_single_stage.json` | 输入阶梯数据 |
| `--output` | `output/params/stage_A_parameters.json` | 输出参数 |
| `--fit-max-step` | `16` | 拟合时包含的最大层数 |
| `--weight-alpha` | `1.0` | 权重公式 w = 1 + α/(n+1) 的 α 值 |
| `--lr` | `0.03` | Adam 学习率 |
| `--steps` | `2000` | 每种颜色的优化步数 |
| `--tol` | `1e-8` | 早停容差 |

---

### Step 3 — 拟合 Stage B (E, k, γ, δ, C₀)

利用多色组合的实测数据（ColorDB），在 Stage A 基础上联合拟合完整模型参数。

```bash
python -m modeling.pipeline.step3_fit_stage_b \
    --db modeling/dbs \
    --stage-a modeling/output/params/stage_A_from_stairs.json \
    --output modeling/output/params/stage_B_retrained.json
```

`--db` 可以是单个 JSON 文件或包含多个 ColorDB JSON 的目录。

**拟合策略（两阶段）：**
1. **Stage 1 (辅助参数预热)：** 冻结 E, k，仅训练 γ、δ、C₀（`--stage1-steps` 步）
2. **Stage 2 (联合优化)：** 解冻所有参数，联合训练（`--steps` 步）

损失函数在 OpenCV Lab 空间计算，同时加入正则化项：
- `--lambda-reg`：防止 E, k 偏离 Stage A 过远
- `--lambda-c0`：防止 C₀ 偏离锚定值过远（默认 2.0，建议范围 1.0~5.0；锚定值来自 Stage A 实测基板颜色）
- `--lambda-neighbor`：约束邻层交互 δ 的 L2 范数
- `--lambda-height-scale`：约束层高缩放 γ 的 L2 范数

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--db` | (必需) | ColorDB JSON 文件或目录 |
| `--stage-a` | `output/params/stage_A_parameters.json` | Stage A 参数 |
| `--output` | `output/params/stage_B_parameters.json` | 输出参数 |
| `--micro-layer-height` | `0.04` | 微层厚度 (mm) |
| `--micro-layer-count` | `20` | 固定微层总数（含填充） |
| `--substrate-mode` | `boundary` | 基板建模方式 (`boundary` / `material`) |
| `--lambda-reg` | `0.1` | E, k 正则化权重 |
| `--lambda-neighbor` | `2.0` | δ 正则化权重 |
| `--lr` | `0.02` | Adam 学习率 |
| `--stage1-steps` | `600` | 预热阶段步数 |
| `--steps` | `800` | 联合优化步数 |
| `--disable-learn-c0` | `False` | 禁用 C₀ 学习 |
| `--disable-height-scale` | `False` | 禁用层高缩放 γ |

---

### Step 4 — 选取代表性配方

使用 k-center 算法在 Lab 色域中选取最大覆盖的代表性配方，用于打印校准色板。

```bash
python -m modeling.pipeline.step4_select_recipes \
    --stage-b modeling/output/params/stage_B_retrained.json \
    --db modeling/dbs \
    --mode 0.04x10 \
    --count 1024 \
    --output modeling/output/recipes/recipes_0p04_10L.json
```

**算法流程：**
1. 从现有 ColorDB 收集已有配方
2. 随机生成候选配方（排除已有的），数量由 `--prefilter-size` 控制
3. 用前向模型预测所有候选配方的 Lab 颜色
4. 运行 k-center 贪心算法：从距色域中心最远的配方开始，每次选取距已选集最远的新配方
5. 输出 `--count` 个代表性配方

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage-b` | `output/params/stage_B_retrained.json` | Stage B 参数 |
| `--db` | (必需) | ColorDB 路径（排除已有配方并获取 palette 信息） |
| `--mode` | `0.04x10` | 层高×层数模式 (`0.04x10` / `0.08x5`) |
| `--count` | `1024` | 选取配方数量 |
| `--prefilter-size` | `50000` | 候选池大小 |
| `--layer-order` | (从 DB 推断) | 打印层序 (`Top2Bottom` / `Bottom2Top`) |
| `--output` | `output/recipes/representative_recipes.json` | 输出路径 |

---

### Step 5 — 构建运行时模型包

将模型参数和大量候选配方的预测 Lab 值打包为 JSON，供 C++ 端的 `image_to_3mf` 程序进行最近邻颜色匹配。

```bash
python -m modeling.pipeline.step5_build_model_package \
    --stage modeling/output/params/stage_B_retrained.json \
    --db modeling/dbs \
    --modes 0.08x5,0.04x10 \
    --candidate-count 65536 \
    --output modeling/output/packages/model_package_phaseA.json
```

模型包中包含多种模式（如 0.08×5 和 0.04×10），每种模式预先计算数万条配方的预测 Lab 值。C++ 端在运行时对目标颜色进行 kd-tree 最近邻查找。

**主要参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage` | (必需) | Stage B 参数 JSON |
| `--db` | (可选) | ColorDB，用于收集种子配方 |
| `--modes` | `0.08x5,0.04x10` | 逗号分隔的模式列表 |
| `--candidate-count` | `65536` | 每种模式的候选配方数 |
| `--threshold` | `5.0` | C++ 端匹配阈值 (DeltaE) |
| `--margin` | `0.7` | C++ 端匹配余量 |
| `--output` | `output/packages/model_package_phaseA.json` | 输出路径 |

---

## 评估工具

### 预览图质量评估

通过 CIEDE2000 色差指标评估渲染预览图与参考照片的匹配程度。

```bash
python -m modeling.eval.preview_quality \
    --data-dir data \
    --preview-dir modeling/output/previews \
    --images xhs1 columbina \
    --modes colordb_only model_only mixed \
    --output-json modeling/output/reports/preview_quality_report.json
```

**评估指标 BADE (Biased Average DeltaE)：**

```
BADE = mean(ΔE₀₀) + w × |chroma_bias|
```

其中 ΔE₀₀ 为 CIEDE2000 色差，`chroma_bias` 为系统性色度偏差，`w` 默认 0.7。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--images` | (必需) | 图像名称列表（不含后缀） |
| `--modes` | `colordb_only model_only mixed` | 评估模式 |
| `--weight-chroma` | `0.7` | 色度偏差惩罚权重 |
| `--output-json` | (可选) | JSON 报告输出路径 |

### 单色通道曲线绘制

绘制单色阶梯数据中各颜色的 RGB/Lab 通道随层数变化的曲线。

```bash
python -m modeling.eval.plot_stage_channels \
    --input modeling/output/params/1_single_stage.json \
    --value-key measured_srgb \
    --x-key step_layers \
    --output-dir modeling/output/reports/plots
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--input` | `output/params/1_single_stage.json` | 阶梯数据 JSON |
| `--value-key` | `measured_srgb` | 绘制字段 (`measured_srgb` / `measured_linear_rgb` / `measured_lab_d65`) |
| `--x-key` | `step_layers` | X 轴 (`step_layers` / `thickness_mm`) |
| `--output-dir` | `output/reports/plots` | 图片输出目录 |
| `--colors` | (全部) | 逗号分隔的颜色名筛选 |
| `--show` | `False` | 交互式显示 |

### Stage A 诊断图

绘制 Stage A 拟合的诊断图，包括 DeltaE 分布、各通道拟合曲线与实测值对比。

```bash
python -m modeling.eval.plot_stage_a_diagnostics \
    --stage-json modeling/output/params/1_single_stage.json \
    --params-json modeling/output/params/stage_A_from_stairs.json \
    --output-dir modeling/output/reports/plots
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage-json` | `output/params/1_single_stage.json` | 原始阶梯数据 |
| `--params-json` | `output/params/stage_A_from_stairs.json` | Stage A 拟合结果 |
| `--output-dir` | `output/reports/plots` | 图片输出目录 |
| `--max-step` | (全部) | 绘图最大层数 |

---

## 核心模块 (core/)

| 模块 | 主要导出 | 功能 |
|------|----------|------|
| **color_space.py** | `srgb_to_linear`, `linear_to_srgb`, `linear_rgb_to_lab_d65`, `linear_rgb_to_opencv_lab_batch`, `lab_grad_from_linear_batch` | 统一的色彩空间转换：sRGB ↔ linear RGB → XYZ → Lab D65 / OpenCV Lab，支持批量和梯度计算 |
| **math_utils.py** | `sigmoid`, `softplus`, `softplus_grad`, `logit`, `softplus_inv`, `round_list` | 激活函数及其逆函数、数值工具 |
| **adam.py** | `adam_optimize(params, loss_fn, ...)` | 通用 Adam 优化器，接受命名参数字典和用户定义的损失+梯度函数 |
| **io_utils.py** | `load_json`, `normalize_label`, `parse_layer_order`, `resolve_db_paths` | JSON 读取、标签规范化（去空格+小写+仅字母数字）、路径解析 |
| **forward_model.py** | `StageForwardModel`, `load_stage_forward_model`, `predict_linear_batch`, `resolve_substrate_idx` | 前向模型的数据结构、参数加载、批量预测 |
| **colorchecker_tool.py** | `ColorCheckerTool` | 从 LabelMe JSON 加载 ColorChecker 参考色块多边形，通过单应矩阵变换到目标图像，提取色块颜色 |
| **color_calibration.py** | `calibrate_image_with_colorchecker`, `apply_ccm_to_image`, `compute_ccm_from_colorchecker` | 基于 ColorChecker 计算 3×3 颜色校正矩阵 (CCM)，应用到图像 |

**在自己的代码中使用核心模块：**

```python
from modeling.core.forward_model import load_stage_forward_model, predict_linear_batch
from modeling.core.color_space import linear_rgb_to_opencv_lab_batch
import numpy as np

# 加载模型
model = load_stage_forward_model("modeling/output/params/stage_B_retrained.json")

# 定义配方: 每行是一个配方，值为颜色索引 (0~7)
recipes = np.array([
    [0, 1, 2, 3, 4, 5, 6, 7, 0, 1],  # 10 层
    [7, 6, 5, 4, 3, 2, 1, 0, 7, 6],
], dtype=np.int32)

# 预测 linear RGB
linear_rgb = predict_linear_batch(
    recipes, model,
    layer_height_mm=0.04,
    micro_layer_height=0.04,
    base_channel_idx=6,        # white
    layer_order="Top2Bottom",
    substrate_idx=0,
)

# 转为 Lab
lab = linear_rgb_to_opencv_lab_batch(linear_rgb)
print(lab)
```

---

## 数据文件说明

### ColorDB 格式

ColorDB 是多色组合实测数据的 JSON 文件，存放于 `dbs/` 目录：

```json
{
  "palette": [
    {"color": "Red", "material": "PLA Basic"},
    {"color": "Yellow", "material": "PLA Basic"},
    ...
  ],
  "layer_height_mm": 0.08,
  "layer_order": "Top2Bottom",
  "base_channel_idx": 7,
  "base_layers": 0,
  "entries": [
    {
      "recipe": [0, 1, 2, 3, 4],
      "lab": [52.3, 18.7, -5.2]
    },
    ...
  ]
}
```

- **palette：** 调色板定义，`color` 字段需与 Stage A 的颜色名匹配（忽略大小写和空格）
- **recipe：** 颜色层序列，值为 palette 中的索引 (0-based)
- **lab：** 实测 Lab 颜色值
- **layer_order：** `Top2Bottom` 表示 recipe[0] 是最顶层，`Bottom2Top` 反之

### 输出文件

| 目录 | 文件 | 说明 |
|------|------|------|
| `output/params/` | `1_single_stage.json` | 单色阶梯提取数据 |
| | `stage_A_from_stairs.json` | Stage A 拟合参数 (E, k) |
| | `stage_B_retrained.json` | Stage B 拟合参数 (E, k, γ, δ, C₀) |
| `output/packages/` | `model_package_phaseA*.json` | 运行时模型包 (5~21 MB)，供 `image_to_3mf` 使用 |
| `output/recipes/` | `recipes_0p04_10L.json` | 代表性配方集，供 `gen_representative_board` 使用 |
| `output/reports/` | `preview_quality_report.json` | 预览质量评估报告 |
| `output/previews/` | `*.3mf`, `*.png` | 生成的 3MF 文件和预览图 |

---

## 依赖项

- **Python** ≥ 3.10
- **NumPy** — 数组运算
- **OpenCV (cv2)** — 图像读取、颜色空间转换、单应变换
- **tqdm** — 进度条（可选，缺失时静默运行）
- **matplotlib** — 绘图（仅 eval 脚本需要）
