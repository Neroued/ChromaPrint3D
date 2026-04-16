# Web Backend 模块索引

## 模块职责

`web/backend/` 是基于 Drogon 的 HTTP 服务，提供 `/api/v1/*` 接口、会话管理、异步任务执行和校准板缓存能力。

## 分层结构

- 进程入口：`web/backend/server_main.cpp`
- 配置层：`web/backend/src/config/`
  - 参数定义与解析：`server_config.h` / `server_config.cpp`
- 基础设施层：`web/backend/src/infrastructure/`
  - 会话：`session_store.*`
  - 任务运行时：`task_runtime.*`
  - 数据加载：`data_repository.*`
  - 校准板缓存：`board_runtime_cache.*`
- 应用层：`web/backend/src/application/`
  - 轻量协调器：`server_facade.*`（持有子服务实例，委托调用）
  - 统一 DTO：`service_result.h`
  - 领域服务：
    - `color_db_service.*` — ColorDB 列表与会话库 CRUD
    - `task_service.*` — 任务列表/查询/删除/artifact 下载
    - `convert_service.*` — 光栅/矢量转换任务提交与请求构建
    - `matting_vectorize_service.*` — 抠图与矢量化任务
    - `recipe_editor_service.*` — 配方编辑器（摘要/候选/替换/生成/预测）
    - `calibration_service.*` — 校准板生成/定位/ColorDB 构建
    - `announcement_service.*` — 公告 CRUD、schema 校验、原子持久化、版本戳
  - 共享内核：
    - `request_parsing.*` — 请求解析与校验工具
    - `json_serialization.*` — JSON 序列化辅助函数
    - `colordb_resolution.*` — ColorDB 解析与通道键归一化
- 表现层：`web/backend/src/presentation/`
  - 路由与请求处理：`api_v1_controller.*`
  - 运行时注册：`backend_runtime.*`
  - CORS 处理：`cors_advice.*`
  - 公告写入鉴权与传输 / body-size 前置 advice：`announcement_auth_advice.*`

## 常见改动落点

| 目标 | 入口文件 |
|---|---|
| 新增 API 路由 | `src/presentation/api_v1_controller.h/.cpp` |
| 新增业务能力 | 对应领域的 `src/application/*_service.{h,cpp}`（非 `server_facade`） |
| 新增服务启动参数 | `src/config/server_config.h/.cpp` + `server_main.cpp` |
| 调整任务并发/TTL/内存预算 | `src/infrastructure/task_runtime.h/.cpp` |
| 调整会话策略与 token 行为 | `src/infrastructure/session_store.h/.cpp` |
| 调整 CORS / cookie 策略 | `src/presentation/cors_advice.*` + `api_v1_controller.*` |

## API 修改建议流程

1. 在对应领域的 `*_service.{h,cpp}` 中定义并实现服务方法。
2. 在 `ServerFacade` 中添加委托方法（如需保持 Controller 不直接依赖子服务）。
3. 在 `ApiV1Controller` 声明路由与处理函数。
4. 复用统一 envelope：`{ok,data}` / `{ok,error}`。
5. 若涉及会话，确认 cookie/header/query 的 token 识别链路不被破坏。
6. 同步更新 [README.md](../../../../README.md) 与 [docs/development.md](../../../development.md) 的接口说明。

## Bambu Studio 预设参数

校准板生成端点（`POST /api/v1/calibration/boards`、`POST /api/v1/calibration/boards/8color`）、定位端点（`POST /api/v1/calibration/locate`）和图像转换端点（`POST /api/v1/convert/raster`、`POST /api/v1/convert/vector`）支持 `nozzle_size` 与 `face_orientation` 参数；转换端点另外支持 `base_layers` 与 `double_sided`。后端在 `convert_service.cpp` / `calibration_service.cpp` 解析后传递给核心库：`face_orientation=facedown` 会触发导出几何绕 Y 轴旋转 180°，`base_layers` 可覆盖底板层数，`double_sided=true` 会启用双面镜像色层，并在预设选择阶段强制使用 `facedown` 预设文件。`flip_y` 仍保持图像/坐标系适配语义，不等价于 `face_orientation`。

## 模型包元数据

`GET /api/v1/model-pack/info` 返回所有已加载模型包的元数据（名称、schema 版本、scope、可用 mode 列表）。前端用此接口判断当前 ColorDB 组合是否有匹配的模型包。实现在 `ServerFacade::GetModelPackInfo()` → `ApiV1Controller::ModelPackInfo`。

## Health 指标

`GET /api/v1/health` 中的 `memory.colordb_pool_bytes` 表示“全局 ColorDB 缓存 + 会话 ColorDB”的总估算体量，由 `ColorDBCache` 与 `SessionStore` 增量维护，读取路径保持 O(1)。它是容量近似值，不是 RSS 的精确拆分。

响应同时带有 `announcements_version`（公告内容的 8 位 FNV-1a 指纹）与 `active_announcement_count`，前端通过 watch 前者变化触发重拉。

## 公告系统

- 路由：
  - `GET /api/v1/announcements`（公开）
  - `POST /api/v1/announcements`（需 `x-announcement-token` 头 + HTTPS + body ≤ 16KB）
  - `DELETE /api/v1/announcements/{id}`（需 token）
- 鉴权：`presentation/announcement_auth_advice.cpp` 前置 advice，未配置 token 时写入路由统一返回 `404 not_found`；非 HTTPS 返回 `403 insecure_transport`；token 用常量时间比较。
- 领域：`application/announcement_service.{h,cpp}` 提供 schema 校验（id 正则、双语至少一种、时间窗口合法）、原子持久化、FNV-1a 版本戳、按严重度排序的有效列表。
- 存储目录：
  - 默认：`${data_dir}/announcements.json`（向后兼容）
  - 独立目录：启动附加 `--announcements-dir <PATH>`，写 `<PATH>/announcements.json`，用于 `read_only: true` 容器的独立可写卷场景。
  - 构造 `AnnouncementService` 时自动 `create_directories`，并做 tmp + rename 原子写。
- Facade：`ServerFacade` 构造时根据 `cfg.announcements_dir` 选择目录，否则回退 `cfg.data_dir`；对外暴露 `ListAnnouncements/UpsertAnnouncement/DeleteAnnouncement`。
- 单测：`web/backend/tests/test_announcement_service.cpp` 覆盖 schema、时间窗口、原子写、版本戳、幂等、跨进程 reload、缺目录自建、独立目录持久化。
- 详细契约与部署：[docs/development.md#543-health](../../../development.md)、[docs/deployment.md#公告系统announcements](../../../deployment.md)。

## 配方编辑器

`POST /api/v1/convert/raster/match-only` 提交 match-only 任务，完成颜色匹配但不生成 3MF。任务完成后通过以下端点进行配方编辑：

- `GET /api/v1/tasks/{id}/recipe-editor/summary`：获取配方摘要（区域列表、唯一配方、调色板）
- `POST /api/v1/tasks/{id}/recipe-editor/alternatives`：查询候选替代配方（支持可选 `recipe_pattern` 参数按层颜色模式过滤）
- `POST /api/v1/tasks/{id}/recipe-editor/replace`：替换指定区域的配方
- `POST /api/v1/tasks/{id}/recipe-editor/generate`：异步生成 3MF 模型
- `POST /api/v1/tasks/{id}/recipe-editor/predict`：使用前向模型预测自定义配方的颜色（请求 `{"recipe":[...]}`，返回 `predicted_lab`/`hex`；无匹配模型时 501）

`region-map` artifact 通过 `GET /api/v1/tasks/{id}/artifacts/region-map` 下载。

约束：仅支持 `dither=None` 的 Raster 流水线，唯一配方数 ≤128。

任务查询语义补充：

- `GET /api/v1/tasks/{id}` 对 `match_only` 转换任务会返回 `generation` 子对象。
- 顶层 `task.status` 只表示主生命周期；若 3MF 生成失败，会呈现为 `status=completed` 且 `generation.status=failed`。
- 只有 `generation.status=succeeded` 时，`result.has_3mf` 才会为 `true`。

核心实现：`task_runtime.h/.cpp`（`SubmitConvertRasterMatchOnly`、`GetRecipeEditorSummary`、`QueryRecipeAlternatives`、`ReplaceRecipe`、`SubmitGenerateModel`、`GetTaskPrintProfile`、`GetTaskColorDbNames`）、`recipe_editor_service.cpp`（`RecipeEditorSummary`、`RecipeEditorAlternatives`、`RecipeEditorReplace`、`RecipeEditorGenerate`、`RecipeEditorPredict`）。

## 前向色彩模型

`DataRepository` 启动时从 `data/models/forward/` 加载 `ForwardModelRegistry`（C++ core 实现），通过 `ForwardModels()` 暴露访问。`RecipeEditorPredict` 根据 task 使用的 ColorDB 确定 vendor/material，调用 `ForwardModelRegistry::Select` 匹配前向模型，将 palette 通道映射为 stage 通道后调用 `ForwardColorModel::PredictLab`。作用域匹配逻辑与 `ModelPackageRegistry::Select` 完全一致。

## 矢量色块宽度分析

`POST /api/v1/convert/vector/analyze-width` 是同步端点，接收 SVG 文件和目标尺寸参数，返回每个色块的面积和最小宽度统计。核心实现在 `core/include/chromaprint3d/shape_width_analyzer.h`，通过距离变换 + 骨架化计算最小宽度。详细参数说明见 [docs/development.md 5.8](../../../development.md)。

## 最小验证

```bash
cmake --build build -j$(nproc)
build/bin/chromaprint3d_server --help
```

本地接口冒烟（服务启动后）：

```bash
curl -s http://127.0.0.1:8080/api/v1/health
```

## 相关任务手册

- [docs/agents/tasks/add_api_endpoint.md](../../tasks/add_api_endpoint.md)
- [docs/agents/tasks/extend_cli_flag.md](../../tasks/extend_cli_flag.md)（当后端参数与 CLI 参数需要联动）

