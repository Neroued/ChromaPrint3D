# 可微物理叠色模型与校准闭环说明（20 微层版本）

本文档把你已经明确的需求整理成一套**可微、物理意义清晰、能用最少测试件测到关键量、并能用 1024 组合数据 + 迭代打印校准**的建模与落地流程。核心目标是：给定一个“多层配方序列”，预测其最终颜色；并通过数据拟合得到每种材料/通道的物理参数，使模型能用于筛选、聚类、代表配方选择与迭代校准。

> 重要原则：**层叠递推在“线性光强空间（linear RGB 或 XYZ）”里完成**；误差评估可以在 Lab（ΔE）里做，但不要在 Lab 里直接递推。

---

## 0. 与现有工具链对齐：你已经有哪些数据/文件

你现有的 ChromaPrint3D 工具链会生成两类关键 JSON（字段名以下均为“真实实现中的键名”，请保持一致）：

### 0.1 `calibration_board.json`（校准板元数据）

由命令行 `gen_calibration_board` 输出的 `meta` 文件。
字段实现位置：`/home/neroued/ChromaPrint3D/core/src/calib/calib.cpp`（`CalibrationBoardMeta::SaveToJson`）。关键字段：

- **`name`**：板名
- **`grid_rows` / `grid_cols`**：色块网格尺寸
- **`patch_recipe_idx`**：长度为 `grid_rows*grid_cols` 的数组（行优先），每个元素是该 patch 对应的“配方索引”。无效值会是一个很大的数（实现里是 `0xFFFF`）。
- **`config`**：配置对象
  - **`config.base_layers`**：底板层数（几何/打印用；光学模型可选择是否显式建模）
  - **`config.base_channel_idx`**：底板使用的通道索引（0..num_channels-1）
  - **底板网格**：导出 3MF 时底板会作为独立网格追加在通道网格之后；若提供 base-only 掩码（如校准板边缘/背景），base_channel 的颜色层也会路由到该底板网格
  - **`config.layer_height_mm`**：打印层高（例如 0.08）
  - **`config.recipe`**：配方空间定义
    - **`config.recipe.num_channels`**：通道数量（2~4）
    - **`config.recipe.color_layers`**：颜色层数（当前固定为 5）
    - **`config.recipe.layer_order`**：层顺序（`"Top2Bottom"` 或 `"Bottom2Top"`）
  - **`config.palette[]`**：通道元数据数组（长度等于 `num_channels`）
    - **`palette[i].color`**：通道名称（例如 `"White"`, `"Yellow"`, `"Red"`, `"Blue"`）
    - **`palette[i].material`**：材料名称（例如 `"PLA Basic"`）

### 0.2 `color_db.json`（1024 组合数据：颜色 ↔ 配方）

由命令行 `build_colordb` 根据拍摄图片 + `calibration_board.json` 构建。
字段实现位置：`/home/neroued/ChromaPrint3D/core/src/colorDB/colorDB.cpp`（`ColorDB::SaveToJson`）。关键字段：

- **`name`**
- **`max_color_layers`**：配方层数上限（== 5）
- **`base_layers` / `base_channel_idx` / `layer_height_mm` / `line_width_mm`**
- **`layer_order`**：`"Top2Bottom"` / `"Bottom2Top"`
- **`palette[]`**：与上面一致（通道名、材料名）
- **`entries[]`**：每个 entry 对应一个配方的测量颜色
  - **`entries[j].lab`**：`[L, a, b]`（OpenCV 的 Lab 转换结果，作为“工作空间”）
  - **`entries[j].recipe`**：长度为 5 的整型数组，每个值是通道索引（0..num_channels-1）

> 这份 `color_db.json` 就是你所说的 “1024 组合数据（4^5）”。每条样本 = `recipe(5层)` + `measured Lab`。

### 0.3 你需要额外补齐/新增的数据（用于 Stage A）

为了让模型参数具有物理可解释性，并且能跨底板/跨厚度泛化，建议新增最少测试件数据（详见第 4 节）：

- 白底/黑底上，每个通道/材料的**单色阶梯**：微层数 \(n=0..16\)（层高 0.04mm）
- 同一拍摄/校准流程下得到每个台阶的颜色（推荐保存为线性 RGB 或 XYZ；至少要能从拍摄数据稳定得到）

---

## 1. 任务定义：模型要做什么

### 1.1 输入（Input）

对每个样本 \(s\)，模型的输入至少包括：

- **配方序列（recipe）**：\(\mathbf{r}_s=[c_1,\dots,c_L]\)，其中 \(L=5\)，\(c_\ell\in\{0,\dots,C-1\}\) 为通道/材料编号（与 `entries[].recipe` 对齐）
- **层顺序（layer_order）**：决定 `recipe` 数组的物理含义（与 `layer_order` 对齐）
  - 当 `layer_order == "Top2Bottom"`：`recipe[0]` 是**最顶层**，`recipe[4]` 是**最底层**
  - 当 `layer_order == "Bottom2Top"`：`recipe[0]` 是**最底层**，`recipe[4]` 是**最顶层**
  - 本文的物理递推默认按“自底向上”进行，因此会先把 `recipe` 统一转换为“底→顶”顺序再展开为微层（见 2.5 的算法）。这一约定也与 ChromaPrint3D 的体素层序一致（注：`/home/neroued/ChromaPrint3D/core/src/geo/geo.cpp` 中有注释：VoxelGrid 层序为自底向上）。
- **层高（layer_height_mm）**：每个物理层厚度（来自 `color_db.json` 或打印设置）
- **底板类型（substrate）**：白底/黑底/或“打印底板通道”，用于给定边界条件颜色（第 2.4）

为了兼容多层高与固定长度输入，还需要引入：

- **微层单位厚度（micro_layer_height_mm）**：建议取 0.04mm
- **固定微层总层数 \(N\)**：例如 20（本文默认 \(N=20\)）

### 1.2 输出（Output）

模型输出预测颜色：

- **\(\hat{\mathbf{C}}_s\in\mathbb{R}^3\)**：在**线性光强空间**（linear RGB 或 XYZ）中的颜色向量
- 可选输出：
  - \(\hat{\mathbf{C}}^{Lab}_s\)：用于计算 ΔE（验收/报告）
  - 每层的中间状态 \(\hat{\mathbf{C}}^{(m)}\)：用于诊断（哪一层导致误差）

### 1.3 模型实现功能（Functions）

你最终需要的功能模块可以拆成 4 类：

- **前向预测（Forward）**：给定配方（含厚度/层高映射）→ 输出预测颜色
- **参数拟合（Fit/Calibrate）**
  - Stage A：单色阶梯数据拟合每个材料的“本征色/遮盖参数”
  - Stage B：1024 组合数据做全局校准与必要的结构修正
- **代表配方选择（Design of Experiments）**：从候选配方中选出“最能提升模型”的一批去打印
- **闭环迭代（Active Learning Loop）**：打印→测量→加入数据→继续拟合→再选配方

### 1.4 输入/输出/参数清单（建议按此组织数据与实现）

#### 输入（每条样本）

| 名称 | 符号/字段 | 类型/形状 | 单位 | 备注 |
| --- | --- | --- | --- | --- |
| 配方（颜色层） | `recipe` / \(\mathbf{r}\) | int\[5] | - | 与 `color_db.json: entries[].recipe` 一致 |
| 通道数 | `num_channels` / \(C\) | int | - | 通常 2~4 |
| 层顺序 | `layer_order` | string | - | `"Top2Bottom"` / `"Bottom2Top"` |
| 打印层高 | `layer_height_mm` / \(h\) | float | mm | `color_db.json: layer_height_mm` |
| 微层单位 | `micro_layer_height_mm` / \(h_u\) | float | mm | 建议 0.04 |
| 固定微层总数 | \(N\) | int | - | 例如 20；需要满足 \(N \ge 5\cdot \mathrm{round}(h/h_u)\) |
| 底板模式 | `substrate_mode` | enum | - | `boundary`（边界条件）或 `material`（当作材料并填充） |
| 底板颜色（边界） | \(\mathbf{C}^{(0)}\) | float\[3] | - | 在 linear RGB/XYZ；白/黑底分别一套 |
| 底板材料编号（填充） | `substrate_channel_idx` | int | - | 例如 `base_channel_idx` 或专门的白/黑底编号 |

#### 输出（每条样本）

| 名称 | 符号 | 类型/形状 | 备注 |
| --- | --- | --- | --- |
| 预测颜色（线性） | \(\hat{\mathbf{C}}\) | float\[3] | linear RGB 或 XYZ |
| 预测颜色（Lab，可选） | \(\hat{\mathbf{C}}^{Lab}\) | float\[3] | 用于 ΔE 报告/阈值 |
| 中间层状态（可选） | \(\hat{\mathbf{C}}^{(m)}\) | float\[N,3] | 诊断用（哪层出错） |

#### 可学习参数（全局）

| 名称 | 符号 | 类型/形状 | 约束/解释 |
| --- | --- | --- | --- |
| 本征色 | \(\mathbf{E}\) | float\[C,3] | \([0,1]\)，厚到饱和的颜色 |
| 遮盖/衰减 | \(\mathbf{k}\) | float\[C,3] | \(>0\)，越大越不透 |
| 邻层修正（可选） | \(\Delta\mathbf{k}\) | float\[C,C,3] | 仅当出现结构性顺序效应时启用，并施加强正则 |

---

## 2. 可微物理模型（20 微层统一离散化）

### 2.1 颜色表示空间（必须线性）

设颜色向量为 \(\mathbf{C}\in\mathbb{R}^3\)，推荐两种工作方式：

- **方案 A（推荐）**：递推在 **linear RGB（sRGB 去 gamma 后）**，损失在 Lab（ΔE 或 L2）
- **方案 B**：递推在 **XYZ**，损失在 Lab

注意：`color_db.json` 里存的是 Lab（OpenCV 工作空间）。若你用它做 Stage B，建议把“模型预测的 linear RGB/XYZ”通过固定的色彩空间变换转到 Lab，再计算损失；不要直接在 Lab 中递推。

### 2.2 单层的物理参数：本征色 + 透过

把每一层材料 \(i\) 看成同时具有：

- **本征漫反射颜色**：\(\mathbf{E}_i\in[0,1]^3\)
  - 直觉：厚到饱和时，这个材料“看起来”的颜色
- **对下方光的透过率（逐通道）**：
\[
\mathbf{T}_i(t)=\exp(-\mathbf{k}_i\,t),\qquad \mathbf{k}_i\in\mathbb{R}_+^3
\]
其中 \(t\) 是厚度（mm），\(\exp\) 按通道逐元素计算。

参数约束（便于可微优化）：

- \(\mathbf{E}_i=\sigma(\mathbf{u}_i)\)（sigmoid 保证 0~1）
- \(\mathbf{k}_i=\mathrm{softplus}(\mathbf{v}_i)\)（softplus 保证正值）

### 2.3 层叠递推（从底到顶，完全可微）

设第 \(m\) 个微层的材料编号为 \(i_m\)，厚度为 \(t_m\)。递推为：
\[
\mathbf{C}^{(m)}=\left(1-\mathbf{T}_{i_m}(t_m)\right)\odot \mathbf{E}_{i_m}
\;+\;\mathbf{T}_{i_m}(t_m)\odot \mathbf{C}^{(m-1)}
\]

- \(\odot\)：逐通道乘
- \(\mathbf{C}^{(0)}\)：边界条件（“底板/更下方”的颜色）

**物理解释**：

- \((1-\mathbf{T})\) 项：这一层自身的漫反射贡献（越不透明越接近 \(\mathbf{E}\)）
- \(\mathbf{T}\) 项：下方颜色透上来的比例（越透明越接近下方）

### 2.4 底板（substrate）的两种建模方式（建议二选一）

你提出的“白底/黑底分开打印”非常关键。底板的处理建议这样做：

- **方式 1（推荐：最稳）**：把底板当作边界条件
  - 对白底：\(\mathbf{C}^{(0)}=\mathbf{C}^{white}_0\)
  - 对黑底：\(\mathbf{C}^{(0)}=\mathbf{C}^{black}_0\)
  - 这些 \(\mathbf{C}_0\) 由“裸底板区域”测得并固定，不参与递推

- **方式 2（需要固定长度时可用）**：把底板当作一种材料 \(S\)
  - 令 \(\mathbf{E}_S\) 为测得的底板颜色，\(\mathbf{k}_S\) 取很大（快速变不透）
  - 将底板填充为若干个微层（见 2.5），然后从 \(\mathbf{C}^{(0)}=\mathbf{0}\)（近似黑）开始递推
  - 这样“底板层数”在模型里有明确含义，并且输入长度固定

> 在“概念与流程优先”的实现里，你可以先用方式 1（更少参数、更稳），等 Stage B 发现“底板厚度/下垫层影响”时再切到方式 2。

### 2.5 20 微层统一离散化（兼容 0.04 / 0.08mm）

定义：

- **微层厚度单位**：\(h_u = 0.04\text{ mm}\)
- **固定微层总数**：\(N=20\)

对任意物理层高 \(h\)（例如 0.04 或 0.08），把“1 个物理层”映射为：
\[
n_u=\mathrm{round}(h/h_u)\quad(\text{例如 }0.04\to 1,\;0.08\to 2)
\]

然后把 5 层配方 \(\mathbf{r}=[c_1,\dots,c_5]\) 展开为微层序列：

- 先把 `recipe` 统一转换为“底→顶”顺序 \(\mathbf{r}_{bt}\)：
  - 若 `layer_order == "Top2Bottom"`：\(\mathbf{r}_{bt}=\mathrm{reverse}(\mathbf{r})\)
  - 若 `layer_order == "Bottom2Top"`：\(\mathbf{r}_{bt}=\mathbf{r}\)
- 再得到长度 \(M=5\cdot n_u\) 的微层序列（每个物理层重复 \(n_u\) 次）：
\[
\text{micro} = [\underbrace{r_{bt}[0],\dots,r_{bt}[0]}_{n_u},\underbrace{r_{bt}[1],\dots,r_{bt}[1]}_{n_u},\dots,\underbrace{r_{bt}[4],\dots,r_{bt}[4]}_{n_u}]
\]
- 若 \(M < N\)，则在**微层序列底部**补齐 \(N-M\) 个底板材料（或底板通道），得到固定长度 \(N\) 的序列：
\[
(i_1,i_2,\dots,i_N)=[\underbrace{S,\dots,S}_{N-M},\text{micro}]
\]
其中 \(i_1\) 是**最底部微层**，\(i_N\) 是**最顶部微层**；递推按 \(m=1\to N\) 自底向上计算。

**小例子（把 5 层配方映射到 20 微层）**：

- 设 `layer_order="Top2Bottom"`，`recipe=[0,1,2,3,0]`，`h=0.08`，`h_u=0.04`，则 `n_u=2`，`M=10`；取 `N=20`，底板材料 `S=0`
- 则 \(\mathbf{r}_{bt}=\mathrm{reverse}(\mathbf{r})=[0,3,2,1,0]\)（底→顶）
- 展开后 `micro=[0,0, 3,3, 2,2, 1,1, 0,0]`
- 最终微层序列 `i=[0]*10 + micro`（长度 20，左侧是底部）

> 若你采用“底板作为边界条件（方式 1）”，固定长度不是必须：可以直接令 \(N=M\)，并用 \(\mathbf{C}^{(0)}\) 作为底板颜色。

> 关键点：补齐发生在“底部”，保证“额外层”物理上表示下方底板/下垫层，而不是加在最上面。

### 2.6 厚度设定（每个微层的 \(t_m\)）

最简单的做法：

- 每个微层厚度 \(t_m=h_u\)（常数）

若你未来需要支持“同一层中不同局部厚度/挤出差异”，可以把 \(t_m\) 作为输入的一部分（例如来自打印工艺估计），但第一版不需要。

---

## 3. 可学习参数与最小可用模型（baseline）

对每个材料/通道 \(i\in\{0,\dots,C-1\}\)，最小可用参数为：

- \(\mathbf{E}_i\in[0,1]^3\)：本征色（3 个自由度）
- \(\mathbf{k}_i\in\mathbb{R}_+^3\)：遮盖/衰减（3 个自由度）

总参数量：\(6C\)（4 通道时仅 24 个标量），非常适合稳健拟合与解释。

### 3.1 可选的“邻层修正”（当 Stage B 出现结构性误差时启用）

如果组合数据出现明显“顺序效应/相邻层相互作用”（例如 AB 与 BA 差异显著），可以在不破坏物理可解释性的情况下加入小修正：

\[
\mathbf{k}_{eff}(i_m,i_{m-1})=\mathbf{k}_{i_m}+\Delta\mathbf{k}_{i_m,i_{m-1}}
\]

其中 \(\Delta\mathbf{k}\)：

- **小表（lookup table）**：参数量 \(C\times C\times 3\)（4 通道时 48 标量），足够小且直观
- 或 **极小 MLP**：输入两层 one-hot/embedding，输出 \(\Delta\mathbf{k}\)（更平滑，但解释性略弱）

> 建议：先跑 baseline（只学 \(\mathbf{E},\mathbf{k}\)），只有当“误差随层顺序系统性偏移”时再加邻层修正。

---

## 4. 最少测试件设计（把“可测量”与“可学习参数”分离）

你的需求是：白/黑分开打印、尽量合并、每色阶梯 0–16 层。推荐测试件如下。

### 4.1 单色阶梯（Stage A 的核心数据）

对每个材料/通道 \(i\)，在两种底板上分别打印阶梯：

- **底板**：白底一块、黑底一块（可以是物理板，也可以是打印底板）
- **微层单位**：0.04mm
- **阶梯台阶**：\(n=0..16\)（共 17 个厚度）
  - \(n=0\)：裸底板参考区（不打印该颜色）
  - \(n>0\)：该颜色打印 \(n\) 个微层（总厚度 \(n\cdot 0.04\) mm）

为什么需要黑底：

- 白底会把“透过到底板再返回”的路径强烈混入，容易让 \(\mathbf{E}\) 与 \(\mathbf{k}\) 纠缠
  - 黑底近似切断下方返回光，能更干净地看到“材料自身贡献随厚度增长”
  - 同一组 \((\mathbf{E}_i,\mathbf{k}_i)\) 同时解释白底与黑底，是非常强的物理一致性验证

### 4.2 阶梯件如何合并（一次打多色）

你可以在同一块底板上放多个“阶梯条带”，每个条带一种颜色：

- 每个台阶建议 10×10mm 或 12×12mm
- ROI 采样取中心区域（例如缩进 10%）减少边缘效应
- 每种颜色旁边可加一个“厚块”（可选但强烈推荐）
  - 厚度 ≥ 1.0mm（例如 0.04×25 微层）
  - 用于确认是否已接近本征色 \(\mathbf{E}_i\)（饱和）

### 4.3 1024 组合板（Stage B 的全局约束）

你已有的 `color_db.json`（从校准板照片生成）可以直接用于 Stage B：

- 输入：`entries[].recipe`（5 层通道序列） + `layer_height_mm` + `layer_order`
- 目标：`entries[].lab`（测得 Lab）

---

## 5. 校准/拟合流程（一步步）

本节给出从“数据采集 → 拟合 → 诊断 → 迭代打印”的可执行流程。建议严格按顺序做，避免用组合数据直接把参数拉到不物理的位置。

### 5.0 从零实现的模块划分（建议）

你可以把工程实现拆成以下可独立验证的模块（先跑通每个模块，再串起来）：

- **数据解析模块**：读取 `color_db.json`（1024 组合），得到 \((recipe, layer_height_mm, layer_order, measured_lab)\)
- **配方展开模块**：把 5 层配方按 2.5 映射为 \(N\) 个微层材料编号 \(i_1..i_N\)
- **前向预测模块**：给定 \(i_1..i_N\)、\((\mathbf{E},\mathbf{k})\)、底板条件 \(\mathbf{C}^{(0)}\) → 递推出 \(\hat{\mathbf{C}}\)
- **颜色空间模块**：把 \(\hat{\mathbf{C}}\) 转成用于损失/报告的空间（Lab/ΔE）
- **拟合模块**：Stage A（阶梯）先拟合 \((\mathbf{E},\mathbf{k})\)，Stage B（组合）再做全局校准（带正则）
- **实验设计模块**：选代表配方（覆盖 + 压测）

### 5.1 前向预测伪代码（自底向上递推）

（以下仅用于说明数据流；你实际实现可以是任意语言/框架）

```python
def forward(micro_layers, E, k, C0, micro_thickness_mm):
    # micro_layers: list[int] length N, bottom->top
    # E: (C,3), k: (C,3), C0: (3,)
    C = C0
    for i in micro_layers:
        T = exp(-k[i] * micro_thickness_mm)  # elementwise
        C = (1 - T) * E[i] + T * C
    return C  # linear RGB or XYZ
```

### Stage 0：摄影测量与取色一致性（强烈建议先做）

目标：让“测得的颜色”尽可能稳定、可复现，否则再好的模型也会被噪声淹没。

建议规范：

- **固定光源与几何**：同一灯位、同一相机高度/角度、关闭自动曝光/自动白平衡
- **加入 ColorChecker**：每次拍摄同画面包含 ColorChecker（或定期拍一次做校正）
- **ROI 规则固定**：每个 patch 取中心 ROI，排除边缘/高光/阴影
- **输出数据**：至少保存
  - 线性 RGB（或 XYZ）
  - 以及用于报告的 Lab

现有工具提示：

- 工作区里已有 `modeling/colorchecker_tool.py` 可以用四角点透视变换后采样色块并输出 Lab（D50/D65 两种版本）。
- ChromaPrint3D 的 `build_colordb` 默认在 OpenCV Lab 工作空间里对每个 patch 做均值（已做 ROI inset），输出到 `color_db.json`。

> 关键是“整条链一致”：你用哪个 Lab 定义、用哪个白点/色彩适配，就保持一致；如果不同来源混用，误差会变成系统偏差。

### Stage A：只用单色阶梯拟合 \((\mathbf{E}_i,\mathbf{k}_i)\)

对每个颜色 \(i\) 拟合目标函数：
\[
\mathcal{L}_A(i)=\sum_{b\in\{white,black\}}\sum_{n=0}^{16} w_n\;\big\|\mathbf{C}^{(b)}_{i,n}-\hat{\mathbf{C}}^{(b)}_{i,n}\big\|^2
\]

- \(\mathbf{C}^{(b)}_{i,n}\)：测得颜色（建议用 linear RGB/XYZ）
- \(\hat{\mathbf{C}}^{(b)}_{i,n}\)：模型预测（n 微层递推）
- \(w_n\)：权重，建议让薄层更大（信息量更高），例如
  - \(w_n=1+\alpha/(n+1)\)

底板条件：

- 如果用“底板作为边界条件”：对每个底板 \(b\)，测得 \(\mathbf{C}_0^{(b)}\)，令 \(\mathbf{C}^{(0)}=\mathbf{C}_0^{(b)}\)
- 如果用“底板作为材料”：把底板也放进序列并固定参数（或一起学）

优化建议：

- 使用 Adam 或 L-BFGS
- 参数化：\(\mathbf{E}=\sigma(\mathbf{u}), \mathbf{k}=\mathrm{softplus}(\mathbf{v})\)
- 评价：在 Lab 上报告 ΔE（但拟合最好仍在 linear space 或 XYZ）

**Stage A 的验收**（强烈建议）：

- 同一套参数能同时拟合白底与黑底阶梯，多数点 ΔE < 2~3
- 如果黑底明显偏差，通常是测量链或底板处理问题，而不是组合拟合能解决的

### Stage B：用 1024 组合数据做全局校准（并锁定物理参数）

输入数据来自 `color_db.json`：

- 每条样本 \(s\)：`recipe_s`（5 层）+ `layer_height_mm` + `layer_order` + 测得 `lab_s`

**Stage C 生成的新校准板说明**：

- 校准板 meta 将直接存每个位置的 `patch_recipes`（不再依赖 `RecipeAt` 计算）
- 生成的 `color_db.json` 结构保持不变，因此 `modeling/fit_stage_B.py` 可直接用于新拍摄数据

底板条件建议（针对 `color_db.json` 的组合板）：

- 组合板通常有打印底板（见 `base_layers` / `base_channel_idx`）。若底板足够厚且近似不透，可以把它当作**边界条件**：
  - 取 \(\mathbf{C}^{(0)} \approx \mathbf{E}_{base\_channel\_idx}\)（用 Stage A 拟合得到的该通道本征色作为底板颜色）
- 若你发现“底板厚度/下垫层”确实影响预测，则把底板当作**材料填充**（方式 2），其厚度可取：
  - \(t_{base} = base\_layers \cdot layer\_height\_{mm}\)
  - 然后在微层序列底部填充对应数量的底板微层（或直接把 \(\mathbf{C}^{(0)}\) 设为更接近实测的底板颜色）

步骤：

1. 把 `recipe_s` 按层高映射为固定长度 \(N=20\) 的微层序列（见 2.5）
2. 用递推得到 \(\hat{\mathbf{C}}_s\)（linear RGB/XYZ）
3. 把 \(\hat{\mathbf{C}}_s\) 转到 Lab，与 `lab_s` 计算误差
4. 最小化带正则的损失：
\[
\mathcal{L}_B=\sum_{s\in\text{recipes}}\big\|\mathbf{y}_s-\hat{\mathbf{y}}_s\big\|^2
\;+\;\lambda\sum_i\big\|\theta_i-\theta_i^{(A)}\big\|^2
\]

- \(\mathbf{y}_s\)：测得颜色（Lab 或你选择的目标空间）
- \(\hat{\mathbf{y}}_s\)：预测颜色（同空间）
- \(\theta_i=(\mathbf{E}_i,\mathbf{k}_i)\)
- \(\theta_i^{(A)}\)：Stage A 拟合得到的参数（物理锚点）
- \(\lambda\)：正则强度（建议从 1e-2~1e0 的量级扫一遍，取泛化最好者）

为什么需要正则：

- 组合数据能“硬拟合”出很多不物理的参数（例如把某色变成极透明但本征色异常），短期误差会下降但泛化与可解释性会崩。
- Stage A 的单色阶梯提供了物理可识别性；Stage B 的作用应是**微调与修正结构偏差**，而不是推翻物理参数。

### Stage B 后的误差诊断（决定是否加邻层修正）

建议做三类诊断图：

- **按层位置分组**：误差是否集中在顶层/底层（说明层序或厚度映射问题）
- **相邻交换对照**：比较 ABxxx vs BAxxx 的误差差异（说明顺序效应）
- **高频交替序列**：A-B-A-B-A 这类序列是否系统性偏差（说明相邻相互作用）

若出现明显结构性误差，再启用 3.1 的 \(\Delta\mathbf{k}\) 邻层修正，并保持其幅度较小（例如加 \(L2\) 正则约束 \(\|\Delta\mathbf{k}\|^2\)）。

---

## 6. 迭代闭环：代表配方 → 打印 → 校准（类似“梯度下降/主动学习”）

你想要的闭环可以做成标准“实验设计 + 继续拟合”的流程。

### 6.1 每轮如何选“代表配方”

从候选配方集合（可以是全空间或你关注的子空间）挑选两类样本混合：

1. **色域覆盖型（coverage）**：在预测 Lab 空间做 k-center / 最远点采样
   - 目的：让你每轮打印的样本覆盖尽可能大的色域，利于筛选/聚类稳定
2. **模型压力测试型（stress）**：专挑模型最容易错的结构
   - 顶层高遮盖 vs 高透明
   - 高频交替 A-B-A-B…
   - 相邻交换对照对（只交换相邻两层）

可选增强（若你愿意做不确定度）：

- 训练多个不同初始化/子采样的模型（ensemble），对同一配方预测取方差，挑“分歧最大”的配方去打印。

### 6.2 新数据加入后的校准方式

- 把新测得样本加入 Stage B 的训练集，继续最小化 \(\mathcal{L}_B\)（保持 Stage A 正则锚点）
- 每轮记录：
  - 训练集/验证集 ΔE 分布（中位数、90 分位）
  - 对压力测试集的误差是否下降
  - 物理参数是否稳定（\(\mathbf{E},\mathbf{k}\) 是否在合理范围内缓慢收敛）

---

## 7. 验收标准（建议）

### 7.1 单色阶梯验收（Stage A）

- 白底与黑底阶梯同时拟合通过：
  - 多数点 ΔE < 2~3
  - 误差随厚度不应出现明显“单调漂移”（否则通常是测量链或厚度映射错误）

### 7.2 组合数据验收（Stage B）

- 在 1024 样本上：
  - 中位数 ΔE 控制在 2~4（取决于你的拍摄噪声与材料稳定性）
  - 90 分位 ΔE 不要出现大面积长尾（长尾通常意味着顺序效应/邻层相互作用）

### 7.3 泛化验收（闭环）

- 新一轮代表配方打印后：
  - 相比上一轮验证误差显著下降
  - 相邻交换对照的系统偏差减少

---

## 8. 附录：从 `color_db.json` 到训练样本的最小数据表（建议你落地时这样组织）

建议把训练数据整理成每行一个样本的表结构（CSV/Parquet/JSON Lines 均可）：

- `recipe`: `[c1,c2,c3,c4,c5]`（来自 `entries[].recipe`）
- `layer_order`: `"Top2Bottom"` / `"Bottom2Top"`（来自 `layer_order`）
- `layer_height_mm`: `0.04` / `0.08` / ...（来自 `layer_height_mm`）
- `num_channels`: `C`（来自 `palette` 长度）
- `measured_lab`: `[L,a,b]`（来自 `entries[].lab`）
- （可选）`measured_rgb_linear` 或 `measured_xyz`：若你在 Stage 0 做了更严格的颜色校正与线性化
- （可选）`substrate_id`: `"white"` / `"black"` / `"base_channel_0"`（用于指定 \(\mathbf{C}^{(0)}\)）

对应的前向输入就是：\((recipe, layer_height_mm, layer_order, substrate)\)；输出就是 \(\hat{\mathbf{C}}\)。

---

## 9. 你可以直接按此文档开始的“最短路径”

1. 先打印并测量白底/黑底的单色阶梯（0..16 微层）
2. 在 linear RGB 或 XYZ 中拟合每个通道的 \((\mathbf{E}_i,\mathbf{k}_i)\)（Stage A）
3. 用 `color_db.json`（1024 组合）做 Stage B 微调，并加“保持物理参数”的正则
4. 若误差有明显顺序效应，再加邻层修正 \(\Delta\mathbf{k}\)
5. 开始闭环：选代表配方→打印→加入数据→继续拟合

---

## 10. 工程注意事项（常见坑与建议默认值）

- **颜色空间一致性（最重要）**：
  - `color_db.json` 的 `entries[].lab` 来自 OpenCV 的 `cvtColor(BGR->Lab)`；如果你用它做损失，请确保“预测值 → Lab”的变换与其一致，否则会引入系统偏差
  - 如果你用 `modeling/colorchecker_tool.py` 走 D50/Lab_d50，同时又用 OpenCV Lab 做组合板损失，务必明确两者不是同一个 Lab 定义；建议全链路统一一种定义
- **模型输出范围**：
  - 只要 \(\mathbf{E}\in[0,1]^3\)、\(\mathbf{T}\in(0,1)^3\)、\(\mathbf{C}^{(0)}\in[0,1]^3\)，递推本质是逐通道的凸组合，\(\hat{\mathbf{C}}\) 会自然落在 \([0,1]^3\)，无需额外 clamp
- **数值稳定**：
  - 计算 \(\mathbf{T}=\exp(-\mathbf{k}t)\) 时，\(\mathbf{k}t\) 过大可能下溢；建议对 \(\mathbf{k}t\) 做上限截断（例如 50）或在实现里用稳定的 `exp` 处理
- **损失函数建议**：
  - Stage A/B 都可以从 L2 开始，但若拍摄噪声/高光导致离群点，建议改用 Huber/Charbonnier（更稳）
  - Stage A 建议薄层权重大（信息量更高），并对黑底/白底同时拟合
- **参数初始化**：
  - \(\mathbf{E}_i\) 可以用“厚块”或阶梯最大厚度的测量颜色做初始化
  - \(\mathbf{k}_i\) 可以从薄层（n=1..3）的上升速度粗估一个量级再作为初值（不需要很准，关键是别极端）
- **层高映射**：
  - 确保 \(N \ge 5\cdot \mathrm{round}(h/h_u)\)。若未来支持更大层高/更多颜色层，要同步增大 \(N\) 或改为变长序列
- **验收与回归**：
  - 每次改动（例如加邻层修正）都要保留一套固定的对照集合：
    - 单色阶梯（白/黑底）
    - 相邻交换对照对（AB vs BA）
    - 高频交替序列（A-B-A-B-A）

