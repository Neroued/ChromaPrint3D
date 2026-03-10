# ChromaPrint3D Frontend

前端采用 `Vue 3 + TypeScript + Vite + Naive UI`，支持 Browser / Electron 双运行时：浏览器模式默认，Electron 壳已可本地开发与调试。

## 常用命令

```bash
cd web/frontend

# 开发模式（默认代理 /api -> localhost:8080）
npm run dev

# 类型检查（单独执行）
npm run typecheck

# 语法/代码检查（默认本地入口，已包含 typecheck）
npm run lint
npm run lint:fix

# 格式化
npm run format
npm run format:check

# 单元测试 + 覆盖率
npm run test

# 生产构建
npm run build
```

## CI 质量门禁

`release-frontend` 工作流会在构建前执行：

- `npm run lint`（已包含 `npm run typecheck`）
- `npm run test`

## 目录分层（重构后）

```text
src/
├── components/              # 视图组件
│   └── param/               # 参数面板子组件
├── composables/             # 通用交互逻辑（异步任务、缩放拖拽、下载等）
│   └── feature/             # 页面级生命周期编排
├── domain/params/           # 参数域模型与构建器
├── runtime/                 # 运行时能力抽象（browser/electron 兼容层）
├── services/                # 业务服务编排
├── stores/                  # Pinia 全局状态
├── api/                     # 按业务域拆分的 API 客户端（convert/matting/vectorize/...）
├── api.ts                   # API 汇总导出（兼容入口）
└── electron.d.ts            # 预留 Electron IPC 类型契约
```

## 分层约束（新增）

- `components/*` 禁止直接依赖 `api/*`，统一通过 `services/*` 或 `composables/*` 调用。
- `services/*` 负责业务编排与跨模块复用；`api/*` 仅负责请求封装与路径定义。
- 新增接口时优先落在 `src/api/<domain>.ts`，并在 `src/api.ts` 汇总导出。

## 运行时能力矩阵（Browser / Electron）

渲染层统一走 `runtime/*`，根据运行环境自动选择 Electron API 或浏览器回退实现：

| 能力                      | Electron（`window.electron`）             | 浏览器回退                                    |
| ------------------------- | ----------------------------------------- | --------------------------------------------- |
| `runtime/env.ts` API 基址 | `env.apiBase`（来自 `preload`）           | `VITE_API_BASE` 或同源 `/api`                 |
| `runtime/storage.ts`      | `storage.getItem / setItem`               | `localStorage`                                |
| `runtime/theme.ts`        | `theme.getSystemDarkMode`                 | `matchMedia('(prefers-color-scheme: dark)')`  |
| `runtime/download.ts`     | `download.openExternal / saveObjectUrlAs` | `window.open` + `fetch/blob` + `<a download>` |
| `runtime/file.ts`         | `file.pickSingleFile`（系统文件选择）     | 隐式 `<input type="file">`                    |

对应契约定义在 `src/electron.d.ts`，Electron 实现在 `web/electron/src/preload.ts`。

## Electron 本地开发（自动启动后端）

当前仓库已提供 `web/electron` 壳工程。本地开发时，Electron 会在启动阶段自动拉起
`chromaprint3d_server`，不需要手动单独启动后端。

### 一次性准备

先在仓库根目录编译后端可执行文件（供 Electron 主进程启动）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target chromaprint3d_server -j
```

然后安装 Electron 开发依赖：

```bash
cd web/electron
npm install
```

### 启动开发环境

```bash
cd web/electron
npm run dev
```

该命令会同时：

- 启动前端 Vite 开发服务（`http://127.0.0.1:5173`）
- 启动 Electron
- 由 Electron 主进程自动拉起后端（默认自动选择 `18080` 起始可用端口）

### 常用环境变量（可选）

- `CHROMAPRINT3D_BACKEND_PATH`：指定后端二进制路径
- `CHROMAPRINT3D_DATA_DIR`：指定后端 `--data` 目录
- `CHROMAPRINT3D_MODEL_PACK_PATH`：指定后端 `--model-pack` 路径
- `CHROMAPRINT3D_BACKEND_PORT`：指定后端优先端口（占用时会自动递增探测）
- `CHROMAPRINT3D_RENDERER_URL`：指定 Electron 加载的前端地址（默认 `http://127.0.0.1:5173`）
- `VITE_UPLOAD_MAX_MB`：浏览器模式上传预校验体积上限（MB，默认 50）
- `VITE_UPLOAD_MAX_PIXELS`：浏览器模式上传预校验像素上限（默认 16777216）

上传限制优先级为：Electron `window.electron.env.uploadMax*` > `VITE_UPLOAD_MAX_*` > 默认值。

### 生产环境可选：ICP备案 / 公安备案信息展示

浏览器模式下，页脚支持按需展示 ICP 备案号和公安备案号；Electron 模式默认不展示。

建议在 `web/frontend/.env.production.local` 中配置（该文件匹配 `*.local`，默认不会提交到 Git）：

```bash
VITE_SITE_ICP_NUMBER=京ICP备12345678号-1
VITE_SITE_ICP_URL=https://beian.miit.gov.cn/
VITE_SITE_PUBLIC_SECURITY_RECORD_NUMBER=京公网安备11010502000001号
VITE_SITE_PUBLIC_SECURITY_RECORD_URL=https://beian.mps.gov.cn/
```

### 常见问题排查

- Electron 启动即报 “Backend binary not found”：
  - 先在仓库根目录执行后端构建命令，确保存在 `build/bin/chromaprint3d_server(.*)`。
- 页面加载正常但接口报错（健康检查失败）：
  - 检查 `data/` 与 `data/model_pack/model_package.json` 是否存在，或通过环境变量指定正确路径。
- 启动时报端口冲突：
  - `5173` 被占用时先释放端口或修改前端 dev 端口；
  - 后端端口可通过 `CHROMAPRINT3D_BACKEND_PORT` 指定起始值。
- `window.electron` 为 `undefined`：
  - 确认是通过 `web/electron` 的 `npm run dev` 启动，而不是仅运行浏览器版 `npm run dev`。

### WSL 下中文显示异常

若在 WSL Ubuntu 下 Electron 页面出现中文方块或字体发虚，可先安装 CJK 字体：

```bash
sudo apt update
sudo apt install -y fonts-noto-cjk fonts-wqy-microhei
```

安装后重启 Electron 进程即可。
