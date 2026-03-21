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
- 应用层：`web/backend/src/application/server_facade.*`
  - 聚合业务流程并返回统一 `ServiceResult`
- 表现层：`web/backend/src/presentation/`
  - 路由与请求处理：`api_v1_controller.*`
  - 运行时注册：`backend_runtime.*`
  - CORS 处理：`cors_advice.*`

## 常见改动落点

| 目标 | 入口文件 |
|---|---|
| 新增 API 路由 | `src/presentation/api_v1_controller.h/.cpp` |
| 新增业务能力 | `src/application/server_facade.h/.cpp` |
| 新增服务启动参数 | `src/config/server_config.h/.cpp` + `server_main.cpp` |
| 调整任务并发/TTL/内存预算 | `src/infrastructure/task_runtime.h/.cpp` |
| 调整会话策略与 token 行为 | `src/infrastructure/session_store.h/.cpp` |
| 调整 CORS / cookie 策略 | `src/presentation/cors_advice.*` + `api_v1_controller.*` |

## API 修改建议流程

1. 在 `ServerFacade` 定义并实现服务方法。
2. 在 `ApiV1Controller` 声明路由与处理函数。
3. 复用统一 envelope：`{ok,data}` / `{ok,error}`。
4. 若涉及会话，确认 cookie/header/query 的 token 识别链路不被破坏。
5. 同步更新 [README.md](../../../../README.md) 与 [docs/development.md](../../../development.md) 的接口说明。

## Bambu Studio 预设参数

校准板生成端点（`POST /api/v1/calibration/boards`、`POST /api/v1/calibration/boards/8color`）、定位端点（`POST /api/v1/calibration/locate`）和图像转换端点（`POST /api/v1/convert/raster`、`POST /api/v1/convert/vector`）支持 `nozzle_size` 与 `face_orientation` 参数；转换端点另外支持 `base_layers` 与 `double_sided`。后端在 `server_facade.cpp` 解析后传递给核心库：`face_orientation=facedown` 会触发导出几何绕 Y 轴旋转 180°，`base_layers` 可覆盖底板层数，`double_sided=true` 会启用双面镜像色层，并在预设选择阶段强制使用 `facedown` 预设文件。`flip_y` 仍保持图像/坐标系适配语义，不等价于 `face_orientation`。

## 配方编辑器

`POST /api/v1/convert/raster/match-only` 提交 match-only 任务，完成颜色匹配但不生成 3MF。任务完成后通过以下端点进行配方编辑：

- `GET /api/v1/tasks/{id}/recipe-editor/summary`：获取配方摘要（区域列表、唯一配方、调色板）
- `POST /api/v1/tasks/{id}/recipe-editor/alternatives`：查询候选替代配方
- `POST /api/v1/tasks/{id}/recipe-editor/replace`：替换指定区域的配方
- `POST /api/v1/tasks/{id}/recipe-editor/generate`：异步生成 3MF 模型

`region-map` artifact 通过 `GET /api/v1/tasks/{id}/artifacts/region-map` 下载。

约束：仅支持 `dither=None` 的 Raster 流水线，唯一配方数 ≤128。

核心实现：`task_runtime.h/.cpp`（`SubmitConvertRasterMatchOnly`、`GetRecipeEditorSummary`、`QueryRecipeAlternatives`、`ReplaceRecipe`、`SubmitGenerateModel`）、`server_facade.cpp`（`RecipeEditorSummary`、`RecipeEditorAlternatives`、`RecipeEditorReplace`、`RecipeEditorGenerate`）。

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

