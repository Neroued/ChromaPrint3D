# ChromaPrint3D 本地开发与调试指南

本文档用于日常开发、联调与排障。  
所有构建步骤（依赖、编译、Docker 构建链路）统一维护在 [docs/build.md](build.md)。

## AGENTS 协作入口

如果你在和 AI agent 协作开发，建议先看：

- [AGENTS.md](../AGENTS.md)（仓库级规则与导航）
- [docs/agents/README.md](agents/README.md)（模块索引）
- [docs/agents/tasks/README.md](agents/tasks/README.md)（高频任务手册）

## 文档边界

- [docs/build.md](build.md)：构建与打包（唯一事实源）
- [docs/development.md](development.md)：开发运行、调试、联调、开发命令
- [docs/deployment.md](deployment.md)：生产部署、网络拓扑、安全与上线流程

## 项目结构概览

```text
ChromaPrint3D/
├── 3dparty/               # 第三方子模块（spdlog、Clipper2、lunasvg、neroued_vectorizer）
├── core/                  # C++ 核心库（颜色匹配、图像处理、3MF 导出）
├── infer/                 # 深度学习推理封装（ONNX Runtime 后端）
├── apps/                  # CLI 工具
├── web/
│   ├── frontend/          # Vue 3 前端
│   ├── backend/           # C++ Drogon HTTP 服务
│   └── electron/          # Electron 桌面壳
├── modeling/              # Python 建模管线
├── scripts/               # 发布、部署、CI 打包、模型下载/上传脚本
├── tools/                 # 辅助工具（npy_to_colordb.py 等）
├── data/                  # 运行时数据（db/model_packs/models 等）
└── docs/                  # 文档
```

## 1. 开发前置

在开始开发前，请先完成：

1. 按 [docs/build.md](build.md) 完成一次可用构建
2. 确认 `build/bin/chromaprint3d_server` 可执行
3. 下载模型文件（需要抠图功能时）

模型下载命令：

```bash
python scripts/download_models.py
```

## 2. 启动本地开发环境（推荐双终端）

### 终端 A：后端服务

```bash
./build/bin/chromaprint3d_server \
  --data ./data \
  --model-pack ./data/model_packs \
  --port 8080 \
  --http-threads 4 \
  --max-tasks 4 \
  --log-level debug
```

说明：

- `--model-pack` 接受目录（自动发现所有 `.msgpack` 文件）或单个 `.msgpack` 文件。默认路径为 `{data}/model_packs/`。
- 开发模式通常不传 `--web`，静态资源由 Vite 提供。
- `--http-threads` 控制 HTTP 线程；`--max-tasks` 控制异步任务线程。
- 生产跨域部署建议额外传入 `--cors-origin <https-origin>` 与 `--require-cors-origin 1`，避免误开放来源。

### 终端 B：前端开发服务

```bash
cd web/frontend
npm run dev
```

默认地址：`http://localhost:5173`  
Vite 会把 `/api` 代理到 `http://localhost:8080`。

## 3. 日常开发命令

### 3.1 前端质量检查

```bash
cd web/frontend
npm run lint
npm run test

# 需要仅排查类型问题时可单独执行
npm run typecheck
```

### 3.2 后端/API 冒烟检查

```bash
curl -s http://127.0.0.1:8080/api/v1/health
curl -s http://127.0.0.1:8080/api/v1/convert/defaults
curl -s http://127.0.0.1:8080/api/v1/vectorize/defaults
curl -s http://127.0.0.1:8080/api/v1/announcements
```

本地联调公告写入（需附 `--announcement-token dev-token --announcement-allow-http`）：

```bash
./scripts/announce.sh list

CHROMAPRINT3D_API_URL=http://127.0.0.1:8080 \
CHROMAPRINT3D_ANNOUNCEMENT_TOKEN=dev-token \
./scripts/announce.sh create test-001 info info '2026-12-31T23:59:59Z' \
  --title-zh '测试公告' --title-en 'Test announcement' \
  --body-zh '这是一条测试公告。' --body-en 'A test announcement.'
```

### 3.3 常用 CLI 调试命令

```bash
# 生成 4 色校准板
./build/bin/gen_calibration_board --channels 4 --out board.3mf --meta board.json

# 从校准板照片构建 ColorDB
./build/bin/build_colordb --image calib_photo.png --meta board.json --out color_db.json

# 图像转 3MF
./build/bin/raster_to_3mf --image input.png --db color_db.json --out output.3mf --preview preview.png
```

> 矢量化 CLI 工具（raster_to_svg、evaluate_svg）已迁移至独立库 [neroued_vectorizer](https://github.com/neroued/neroued_vectorizer)。

### 3.4 测试运行

如已在构建阶段启用 `CHROMAPRINT3D_BUILD_TESTS=ON`，可执行：

```bash
ctest --test-dir build --output-on-failure
```

### 3.5 C++ 格式检查

CI 会对所有变更的 C++ 文件执行 `clang-format-18` 格式检查（见 `.clang-format` 配置），未通过时 PR 会被阻止合并。建议在本地提交前对修改的文件运行格式化：

```bash
# 安装（Ubuntu / Debian）
sudo apt install clang-format-18

# 对修改的文件执行格式化
clang-format -i <修改的文件>
```

## 4. Electron 本地开发

Electron 主进程会自动拉起后端（要求后端二进制已构建完成）。

```bash
cd web/electron
npm install
npm run dev
```

常用环境变量：

- `CHROMAPRINT3D_BACKEND_PATH`：指定后端二进制路径
- `CHROMAPRINT3D_DATA_DIR`：指定后端 `--data` 目录（打包态默认使用 `<userData>/data`）
- `CHROMAPRINT3D_MODEL_PACK_PATH`：指定后端 `--model-pack` 路径（目录或单个 `.msgpack` 文件）
- `CHROMAPRINT3D_BACKEND_PORT`：指定后端优先端口
- `CHROMAPRINT3D_RENDERER_URL`：指定前端地址（默认 `http://127.0.0.1:5173`）
- `VITE_UPLOAD_MAX_MB` / `VITE_UPLOAD_MAX_PIXELS`：前端上传预校验上限
- `CHROMAPRINT3D_MODEL_DOWNLOAD_RETRIES` / `CHROMAPRINT3D_MODEL_DOWNLOAD_TIMEOUT_MS` / `CHROMAPRINT3D_MODEL_DOWNLOAD_BACKOFF_MS`：Electron 模型下载重试与超时控制

## 5. Web/API 契约（开发关注）

### 5.1 统一响应格式

- JSON 成功：`{ "ok": true, "data": ... }`
- JSON 失败：`{ "ok": false, "error": { "code": "...", "message": "..." } }`
- 二进制下载：直接返回 bytes，不走 envelope

### 5.2 会话 token 识别优先级

1. `session` cookie
2. `X-ChromaPrint3D-Session` header

### 5.3 Health 端点响应格式

`GET /api/v1/health` 响应包含 `memory` 对象以及公告版本戳，提供服务端内存状态与公告变更指示：

```json
{
  "ok": true,
  "data": {
    "status": "ok",
    "version": "1.3.1",
    "active_tasks": 2,
    "total_tasks": 15,
    "memory": {
      "rss_bytes": 1073741824,
      "heap_allocated_bytes": 536870912,
      "heap_resident_bytes": 805306368,
      "artifact_budget_bytes": 134217728,
      "artifact_budget_limit_bytes": 536870912,
      "colordb_pool_bytes": 52428800,
      "memory_limit_bytes": 4294967296,
      "allocator": "jemalloc"
    },
    "announcements_version": "39bd1beb",
    "active_announcement_count": 1
  }
}
```

| 字段 | 说明 |
|------|------|
| `rss_bytes` | 进程 RSS |
| `heap_allocated_bytes` | 应用层已分配的堆内存 |
| `heap_resident_bytes` | 分配器保留的物理内存 |
| `artifact_budget_bytes` | 任务独占数据（recipe_map/region_map 等）占用 |
| `artifact_budget_limit_bytes` | `--max-result-mb` 对应的预算上限 |
| `colordb_pool_bytes` | ColorDB 缓存 + 会话 DB 的总估算体量 |
| `memory_limit_bytes` | 容器内存限制（cgroup）或物理内存总量 |
| `allocator` | 编译时链接的分配器（`jemalloc` / `glibc` / `system`） |
| `announcements_version` | 公告内容的 8 位 hex 指纹（FNV-1a）；变化表示公告增删改，前端据此重新拉取 |
| `active_announcement_count` | 当前有效公告条数（按 `starts_at` ≤ now < `ends_at` 过滤） |

补充说明：

- `colordb_pool_bytes` 是启发式容量指标，不是 RSS 的拆分项。它由全局 `ColorDBCache` 与会话 `SessionStore` 的估算字节数增量维护，`GET /api/v1/health` 读取该值不再做全量遍历，保持 O(1) 热路径。
- `status` 当前固定返回 `ok`；前端应以请求是否成功判断“在线/离线”，不要把 `status === "ok"` 当成唯一在线判据，以便后续扩展 `degraded` 等服务状态。

### 5.4 关键接口分组

| 分组 | 代表接口 |
|---|---|
| 健康与默认参数 | `GET /api/v1/health`、`GET /api/v1/convert/defaults`、`GET /api/v1/vectorize/defaults`、`GET /api/v1/model-pack/info` |
| 数据库管理 | `GET /api/v1/databases`、`GET /api/v1/session/databases`、`POST /api/v1/session/databases/upload` |
| 异步任务提交 | `POST /api/v1/convert/raster`、`POST /api/v1/convert/raster/match-only`、`POST /api/v1/convert/vector`、`POST /api/v1/convert/vector/match-only`、`POST /api/v1/matting/tasks`、`POST /api/v1/vectorize/tasks` |
| 矢量分析 | `POST /api/v1/convert/vector/analyze-width` |
| 配方编辑器 | `GET /api/v1/tasks/{id}/recipe-editor/summary`、`POST /api/v1/tasks/{id}/recipe-editor/alternatives`、`POST /api/v1/tasks/{id}/recipe-editor/replace`、`POST /api/v1/tasks/{id}/recipe-editor/generate`、`POST /api/v1/tasks/{id}/recipe-editor/predict` |
| 任务查询与产物 | `GET /api/v1/tasks`、`GET /api/v1/tasks/{id}`、`GET /api/v1/tasks/{id}/artifacts/{artifact}` |
| 校准链路 | `POST /api/v1/calibration/boards`、`POST /api/v1/calibration/boards/8color`、`POST /api/v1/calibration/locate`、`POST /api/v1/calibration/colordb` |
| 公告 | `GET /api/v1/announcements`（公开）、`POST /api/v1/announcements`（需 token）、`DELETE /api/v1/announcements/{id}`（需 token） |

### 5.4.1 公告 API 契约

- 写入身份：`x-announcement-token` 头 + HTTPS；未配置 token 时 POST/DELETE 一律返回 `404 not_found`（避免向外暴露功能存在性）。
- 传输：生产环境写入必须走 HTTPS（通过 TLS 或反代带 `X-Forwarded-Proto: https`），否则返回 `403 insecure_transport`。
  本地联调可启动时附加 `--announcement-allow-http` 临时放宽。
- Body 上限：`POST` 请求体 ≤ 16 KiB，否则返回 `413 payload_too_large`。
- Schema：
  - `id`：`^[a-zA-Z0-9_-]{1,64}$`，同 `id` 重复 POST 即等同于更新。
  - `type`：`info` / `warning` / `maintenance`
  - `severity`：`info` / `warning` / `error`（决定前端 banner 配色）
  - `title` / `body`：对象形式 `{ "zh": "...", "en": "..." }`，两语言至少一项非空，单语言 ≤ 2000 字符。**纯文本**，前端禁用 `v-html`，URL 不自动链接化。
  - `starts_at`（可选）、`ends_at`（必填）、`scheduled_update_at`（可选）：全部 ISO8601 UTC，均与当前时间字符串比较，无需客户端解析时区。
  - `dismissible`：默认 `true`；`false` 时前端不展示"不再提醒"按钮。
- 成功响应：POST / DELETE / GET 均返回 `announcements_version`（与 health 同步），便于前端局部更新。
- 持久化：`${data_dir}/announcements.json`，原子写（临时文件 rename）。
- 有关部署与 token 分发策略，见 [docs/deployment.md#公告系统announcements](deployment.md#公告系统announcements)。

### 5.4 Bambu Studio 预设参数

以下参数适用于 Raster/Vector 转换端点和校准板生成端点：

| 参数 | 类型 | 默认值 | 有效值 | 说明 |
|---|---|---|---|---|
| `nozzle_size` | string | `"n04"` | `"n02"`, `"n04"`, `"0.2"`, `"0.4"` | 喷嘴尺寸 |
| `face_orientation` | string | `"faceup"` | `"faceup"`, `"facedown"` | 观赏面朝向；`facedown` 时模型导出几何绕 Y 轴旋转 180° |
| `base_layers` | int / null | `null` | `-1`（CLI 语义）或 `>=0` | 底板层数覆盖。API 不传表示继承 ColorDB；CLI 中 `-1` 表示继承，`0` 表示无底板 |
| `double_sided` | bool | `false` | `true`, `false` | 双面生成开关。开启后在底板两侧生成镜像颜色层 |
| `transparent_layer_mm` | float | `0` | `0`, `0.04`, `0.08` | 透明镀层厚度（mm）。仅在 `face_orientation=facedown` 且 `double_sided=false` 时可用。启用后在打印模型最底层追加一层透明材料网格 |
| `gradient_pixel_mm` | float | `0` | `>=0` | 渐变色区域栅格化分辨率（mm/px）。`0`=自动（3×`tessellation_tolerance_mm`）。SVG 中的渐变填充会被栅格化并拆分为纯色子区域参与颜色匹配与 3MF 生成 |

组合后会先选择 `data/presets/` 下 4 个预设之一（如 `bambu_p2s_0.08mm_n04_faceup.json`）。预设中的耗材丝配置（颜色、类型、温度等）完整写入 3MF，Bambu Studio 打开时优先加载文件内配置。模型颜色自动匹配到预设中最接近的耗材丝槽位（RGB 欧氏距离）。此外，`face_orientation=facedown` 会在导出阶段将模型几何整体绕 Y 轴旋转 180°，`faceup` 则保持原方向。

注意：`face_orientation` 与 `flip_y` 语义不同。`flip_y` 用于图像/坐标系方向适配（预处理与体素构建阶段），`face_orientation` 用于最终导出几何朝向控制。

另外，`base_layers` 与 `double_sided` 的优先级高于 ColorDB 默认值：当请求显式传入时，转换链路会按请求值构建模型；不传则沿用 ColorDB 中记录的默认底板配置。

当 `double_sided=true` 时，预设选择会强制使用 `face_orientation=facedown` 对应的 Bambu 预设文件，以保证双面模型的分层参数与预设策略一致。

前端参数面板在开启 `double_sided` 后会自动将 `face_orientation` 固定为 `facedown`，并禁用观赏面朝向选项；关闭双面时会恢复到开启双面前记录的朝向。

`transparent_layer_mm` 用于在 FaceDown 模式下追加一层透明材料网格（"Transparent Layer"），该网格作为独立对象导出到 3MF，覆盖模型的完整 XY 投影范围。用户在 Bambu Studio 中将该对象指定为透明耗材即可获得透亮效果。该参数在 `face_orientation=faceup` 或 `double_sided=true` 时被忽略/禁止。前端在不满足条件时自动隐藏该选项并重置为 0。

### 5.5 Raster 匹配参数说明

- `cluster_method`：`kmeans`（默认）或 `slic`
- `kmeans` 路径参数：`cluster_count`（默认 `0`）
  - `0`：自动检测最优聚类数
  - `1`：逐像素匹配（不聚类）
  - `>= 2`：手动指定聚类数
- `slic` 路径参数：`slic_target_superpixels`、`slic_compactness`、`slic_iterations`、`slic_min_region_ratio`
- 互斥规则：当 `cluster_method=slic` 时，后端会强制 `dither=none`

### 5.6 Vectorize 透明 PNG 行为

- 输入为带 alpha 的 PNG 时，`alpha==0` 的像素不会参与矢量化，导出的 SVG 默认保持透明背景。

### 5.7 Vectorize 参数说明

以下参数用于 `POST /api/v1/vectorize/tasks` 的 `params` JSON，同时 `GET /api/v1/vectorize/defaults`
会返回对应默认值。

| 参数 | 类型 | 默认值 | 有效范围 | 说明 | 建议暴露层级 |
|---|---|---:|---|---|---|
| `num_colors` | int | 0 | `0` 或 `[2,256]` | 颜色量化数量。`0`=自动检测最优数量；`2-256`=显式指定 | 前端主参数 |
| `curve_fit_error` | float | 0.8 | `[0.1,5]` | Bezier 拟合误差阈值（像素），越小越贴边界 | 前端主参数 |
| `contour_simplify` | float | 0.45 | `[0,10]` | 轮廓简化强度，越大节点越少 | 前端主参数 |
| `svg_enable_stroke` | bool | true | `true/false` | 启用细线增强描边输出 | 前端主参数 |
| `enable_coverage_fix` | bool | true | `true/false` | 启用覆盖修复补洞 | 前端主参数 |
| `smoothing_spatial` | float | 15 | `[0,50]` | Mean Shift 空间窗口（sp） | 前端高级参数 |
| `smoothing_color` | float | 25 | `[0,80]` | Mean Shift 颜色窗口（sr） | 前端高级参数 |
| `thin_line_max_radius` | float | 2.5 | `[0.5,10]` | 细线检测距离阈值（像素） | 前端高级参数 |
| `slic_region_size` | int | 20 | `[0,100]` | SLIC 超像素尺寸，越小越精细但更慢 | 前端高级参数 |
| `edge_sensitivity` | float | 0.8 | `[0,1]` | 边缘对齐灵敏度：SLIC 在边缘处降低空间权重的力度 | 前端高级参数 |
| `refine_passes` | int | 6 | `[0,20]` | 边界像素逐像素细化迭代次数，`0` 禁用 | 前端高级参数 |
| `max_merge_color_dist` | float | 150 | `[0,2000]` | 小区域合并最大 LAB ΔE²，超过则拒绝合并 | 前端高级参数 |
| `min_region_area` | int | 10 | `>=0` | 小区域合并阈值（像素²） | 前端高级参数 |
| `min_contour_area` | float | 10.0 | `>=0` | 最小保留轮廓面积（像素²） | 前端高级参数 |
| `min_hole_area` | float | 4.0 | `>=0` | 最小保留孔洞面积（像素²） | 前端高级参数 |
| `min_coverage_ratio` | float | 0.998 | `[0,1]` | 覆盖率低于该值时触发 coverage fix | 前端高级参数 |
| `svg_stroke_width` | float | 0.5 | `[0,20]` | SVG 描边宽度（像素） | 前端高级参数 |
| `smoothness` | float | 0.5 | `[0,1]` | 轮廓平滑度：0 = 保留所有细节，1 = 最大平滑 | 前端主参数 |
| `detail_level` | float | -1 | `[-1,1]` | 统一细节控制：`>=0` 时自动推导 `curve_fit_error` 与 `contour_simplify`；`-1` 禁用 | 前端主参数 |
| `merge_segment_tolerance` | float | 0.05 | `[0,0.5]` | 近线性 Bezier 段合并容差（弦长比），`0` 禁用合并 | 前端高级参数 |
| `enable_antialias_detect` | bool | false | `true/false` | 启用抗锯齿混合边缘检测，改善 AA 源图的边界精度 | 前端高级参数 |
| `aa_tolerance` | float | 10 | `[1,50]` | AA 混合像素检测 LAB Delta-E 容差 | 仅 CLI / 调试 |
| `corner_angle_threshold` | float | 135 | `[90,175]` | 角点判定阈值（度） | 仅 CLI / 调试 |
| `upscale_short_edge` | int | 600 | `[0,2000]` | 自动放大触发短边阈值，`0` 禁用放大 | 仅 CLI / 调试 |
| `max_working_pixels` | int | 3000000 | `[0,100000000]` | 大图预处理像素上限；超出时先按面积缩小以控制 SVG 复杂度，`0` 禁用 | 仅 API / 调试 |

说明：

- 前端默认选择性暴露：主参数 + 高级参数；技术性参数优先在 CLI 调试。
- `num_colors=0`（默认）时，管线通过复合评分候选搜索自动选择最优颜色数。评估重建误差、碎片化代价和复杂度惩罚，对候选 K∈{2,3,4,6,8,12,16,24} 取 argmin。结果通过 `VectorizeResult.resolved_num_colors` 返回。
- 建议先调整 `num_colors`、`smoothness`、`detail_level`，它们能覆盖大部分质量/复杂度权衡。
- `detail_level` 启用时（`>=0`）会自动设置 `curve_fit_error` 与 `contour_simplify`，无需手动调。
- 当源图分辨率很高（如 4K+）且导出 SVG 过大时，优先保留 `max_working_pixels` 默认值或适当下调。
- 对含抗锯齿边缘的源图（如 Photoshop 导出），可启用 `enable_antialias_detect` 以获取更准确的边界。

#### 边缘感知分割与逐像素标签细化

多色模式下的分割流水线包含以下优化，可通过上述参数调节：

1. **边缘感知 SLIC**（`edge_sensitivity`）：预处理阶段从未平滑图像计算梯度边缘图，在 SLIC 超像素聚类时降低边缘处的空间权重，使超像素边界自动对齐图像中的颜色边缘。
2. **逐像素标签细化**（`refine_passes`）：SLIC + K-Means 分割后，对标签边界像素做迭代修正（使用未平滑的原始颜色信息），将被超像素粒度错误归类的像素重新分配到颜色最接近的标签。此步骤改善了细线保留和色块边界精度。
3. **对比度感知小区域合并**（`max_merge_color_dist`）：`MergeSmallComponents` 在合并小区域时会检查颜色距离，当小区域的颜色与合并目标差异过大时拒绝合并，避免高对比度细特征被强制吞并。

### 5.8 矢量色块宽度分析

`POST /api/v1/convert/vector/analyze-width`（同步、multipart）

对 SVG 中每个色块的最小宽度进行分析，帮助用户在转换前确定合适的缩放比例。

**请求字段：**

| 字段 | 类型 | 说明 |
|---|---|---|
| `svg` | file | SVG 文件 |
| `params` | JSON string | 分析参数 |

**params 参数：**

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `target_width_mm` | float | `0` | 目标宽度（mm），0 表示使用 SVG 原始尺寸 |
| `target_height_mm` | float | `0` | 目标高度（mm），0 表示使用 SVG 原始尺寸 |
| `min_area_mm2` | float | `1.0` | 面积过滤阈值（mm²），小于此值的色块跳过分析 |
| `raster_px_per_mm` | float | `20.0` | 光栅化分辨率（px/mm），影响精度 |
| `flip_y` | bool | `false` | 是否翻转 Y 轴 |

**响应示例：**

```json
{
  "ok": true,
  "data": {
    "image_width_mm": 100.0,
    "image_height_mm": 80.0,
    "total_shapes": 42,
    "filtered_count": 5,
    "min_area_threshold_mm2": 1.0,
    "shapes": [
      { "index": 0, "color": "#FF0000", "area_mm2": 12.3, "min_width_mm": 0.35, "median_width_mm": 1.2 }
    ]
  }
}
```

算法说明：对每个通过面积筛选的色块，按 bounding box 光栅化为二值 mask，做距离变换和 Zhang-Suen 骨架化，骨架上距离变换最小值 x2 即为该色块最小宽度。

### 5.9 Matting 后处理参数

`POST /api/v1/matting/tasks/{id}/postprocess` 支持以下 JSON 字段：

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---:|---|
| `threshold` | float | `0.5` | alpha 二值化阈值（`[0,1]`） |
| `morph_close_size` | int | `0` | 闭运算核大小，`0` 表示关闭 |
| `morph_close_iterations` | int | `1` | 闭运算迭代次数 |
| `min_region_area` | int | `0` | 过滤小连通域阈值（像素） |
| `outline.enabled` | bool | `false` | 是否生成描边图层 |
| `outline.width` | int | `2` | 描边宽度（像素，超大图会自适应放大） |
| `outline.color` | `[b,g,r]` | `[0,0,255]` | 描边颜色（BGR） |
| `outline.mode` | string | `center` | `center` / `inner` / `outer` |
| `reframe.enabled` | bool | `false` | 是否重构画布（按前景 bbox 裁剪空白并留边） |
| `reframe.padding_px` | int | `16` | 外边缘留白像素（`[0,4096]`） |

行为说明：

- 当 `reframe.enabled=true` 时，后处理输出尺寸会随前景区域变化，不再固定等于原图尺寸。
- 描边/闭运算会在安全边界内计算，降低前景贴边时被截断的概率。

### 5.10 任务系统生命周期

- 任务状态：`pending -> running -> completed/failed`
- 队列上限：`--max-queue`
- 单会话并发上限：`--max-owner-tasks`
- 结果缓存 TTL：`--task-ttl`
- 结果总内存预算：`--max-result-mb`

对 `match_only` 转换任务还有一层独立的生成子状态：

- 配色匹配阶段完成后，顶层 `task.status` 进入 `completed`，此时配方编辑器所需的 summary / region-map / preview 已可用。
- `POST /api/v1/tasks/{id}/recipe-editor/generate` 只推进 `generation.status`：`idle -> running -> succeeded/failed`。
- 当 `generation.status=failed` 时，顶层 `task.status` 仍保持 `completed`，以保留 recipe editor 的可编辑与可重试语义；此时 `result.has_3mf=false`。
- 只有 `generation.status=succeeded` 后，`result.has_3mf` 才会变为 `true`，3MF 下载端点才可用。

`GET /api/v1/tasks/{id}` 在 `match_only` 转换任务上会返回：

```json
{
  "ok": true,
  "data": {
    "id": "task-id",
    "status": "completed",
    "match_only": true,
    "generation": {
      "status": "failed",
      "error": "Cannot open 3MF spill file: ..."
    },
    "result": {
      "has_3mf": false,
      "has_preview": true,
      "has_source_mask": true
    }
  }
}
```

保活语义：

- 后端内部将任务终态时间与访问时间拆分为 `completed_at` 和 `last_accessed_at`。
- `completed_at` 只在任务主生命周期真正结束时写入，用于 `--task-ttl`。
- `GET /api/v1/tasks/{id}`、artifact 下载、recipe editor 查询只会刷新 `last_accessed_at`，不会延长 `--task-ttl`。

## 6. 常见开发问题

### 6.1 前端启动了但 API 全失败

- 检查后端是否监听在 `8080`
- 检查 `chromaprint3d_server` 启动日志是否报错

### 6.2 Electron 启动时报后端缺失

- 先按 [docs/build.md](build.md) 完成后端二进制构建
- 或通过 `CHROMAPRINT3D_BACKEND_PATH` 显式指定路径

### 6.3 上传图片时报大小/像素限制

- 后端限制：`--max-upload-mb`、`--max-pixels`
- 前端预检限制：`VITE_UPLOAD_MAX_MB`、`VITE_UPLOAD_MAX_PIXELS`

### 6.4 子模块目录为空

```bash
git submodule update --init --recursive
```

## 7. 模型管理

模型文件（`.onnx`）不随仓库直接分发，按清单动态下载。

### 7.1 下载模型

```bash
python scripts/download_models.py
```

说明：

- 开发脚本默认从 `data/models/models.json` 的 `base_url` 下载。
- Electron 应用内下载优先使用 `models.json` 的 `sources`（可配置多源回退），并下载到用户可写目录（默认 `<userData>/data/models`）。
- Electron 下载前支持连通性检查（按源探测可达性），建议先检查再下载。
- Electron 下载完成后需要重启应用，后端会在启动阶段重新发现并加载模型。

### 7.2 上传/更新模型（维护者）

```bash
pip install huggingface_hub
huggingface-cli login
python scripts/upload_models.py --create-repo
```

### 7.3 同步到国内源（ModelScope，维护者）

1) 在 ModelScope 创建仓库（建议与 HF 同名），并保持目录结构与 `models.json` 的 `models[].path` 一致。  
2) 使用 Git LFS 推送模型文件：

```bash
git lfs install
git clone "https://www.modelscope.cn/models/<org>/<repo>.git"
cd <repo>
mkdir -p matting
cp /path/to/*.onnx matting/
git add matting/*.onnx
git commit -m "update model files"
git push
```

3) 更新本仓库 `data/models/models.json` 的 `sources`（优先国内源），并同步 `sha256/size_bytes`。

## 8. 发布流程

发布通过 PR 驱动，版本 tag 由 CI 自动创建。支持**预览发布**（pre-release）和**正式发布**两种模式。

### 8.1 预览发布（推荐先行）

预览版用于在正式发布前进行部署级测试。版本格式为 `X.Y.Z-rc.N`（release candidate）。

```bash
# 自动从 rc.1 开始（或从已有 rc tag 递增）
./scripts/release-preview.sh 1.2.12

# 显式指定 rc 编号
./scripts/release-preview.sh 1.2.12 2
```

脚本会：

1. 校验版本号格式、自动计算 rc 编号
2. 从最新 `origin/master` 创建 `preview/v1.2.12-rc.1` 分支
3. 更新 `CMakeLists.txt`（含 `CHROMAPRINT3D_VERSION_PRERELEASE`）、`web/frontend/package.json`、`web/electron/package.json`
4. 提交、推送分支、通过 `gh` 创建 PR

合并后 CI 自动打 tag `v1.2.12-rc.1`，触发完整构建流水线：

- GitHub Release 标记为 **Pre-release**
- Docker 镜像打 `preview` / `preview-api` 浮动标签和 `1.2.12-rc.1` / `1.2.12-rc.1-api` 固定标签

如果预览版发现 bug，修复后再执行 `release-preview.sh 1.2.12` 即可自动递增到 `rc.2`、`rc.3`...

### 8.2 正式发布（从预览转正）

```bash
./scripts/release.sh 1.2.12
```

脚本会：

1. 校验版本号格式与连续性（仅对比稳定版 tag）
2. 从最新 `origin/master` 创建 `release/v1.2.12` 分支
3. 更新版本号并**清除预发布后缀**（`CHROMAPRINT3D_VERSION_PRERELEASE` 置空）
4. 提交、推送分支、通过 `gh` 自动创建 PR

### 8.3 合并与发布

1. CI 自动在 PR 上运行（前端 lint/test/build + 后端构建 + 单元测试 + clang-format 检查）
2. 审查通过后合并 PR
3. `release-tag.yml` 工作流检测到 `release/v*` 或 `preview/v*` 分支合并后，自动创建 annotated tag 并推送
4. tag 推送触发 `release.yml`：构建各平台后端/前端/Electron 安装包、Docker 镜像、GitHub Release

### 8.4 完整发布工作流示例

```
release-preview.sh 1.2.12       → preview/v1.2.12-rc.1 → PR → 合并 → tag v1.2.12-rc.1
  ↓ 部署到预览环境测试
  ↓ 发现 bug，修复后...
release-preview.sh 1.2.12       → preview/v1.2.12-rc.2 → PR → 合并 → tag v1.2.12-rc.2
  ↓ 测试通过
release.sh 1.2.12               → release/v1.2.12      → PR → 合并 → tag v1.2.12
```

### 8.5 版本号位置

| 文件 | 用途 |
|---|---|
| `CMakeLists.txt` | C++ 库版本（`project(... VERSION X.Y.Z)` + `CHROMAPRINT3D_VERSION_PRERELEASE`） |
| `web/frontend/package.json` | 前端版本 |
| `web/electron/package.json` | Electron 版本 |

三个文件的版本号由 `release.sh` / `release-preview.sh` 统一更新，无需手动同步。

### 8.6 Docker 镜像标签

| 标签 | 触发条件 | 用途 |
|---|---|---|
| `latest` / `vX.Y.Z` / `vX.Y` | 正式 tag | 一体镜像，开箱即用 |
| `api` / `vX.Y.Z-api` / `vX.Y-api` | 正式 tag | 仅 API，分体部署 |
| `preview` / `vX.Y.Z-rc.N` | 预览 tag | 一体预览镜像 |
| `preview-api` / `vX.Y.Z-rc.N-api` | 预览 tag | 仅 API 预览镜像 |
