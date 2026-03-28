# ChromaPrint3D

[English](#english) | [中文](#中文)

---

<a id="中文"></a>

## 简介

ChromaPrint3D 可以把图片转换成多色 3D 打印可用的 3MF 模型。  
支持 Bambu Studio 预设参数自动写入（0.2mm/0.4mm 喷嘴 × 观赏面朝上/朝下，共 4 种预设），并自动将模型颜色匹配到最接近的耗材丝槽位。  
支持在转换时显式配置底板层数（`base_layers`）、双面生成（`double_sided`）以及透明镀层（`transparent_layer_mm`，仅观赏面朝下模式）。
如果你只是想先体验效果，不需要先读一堆文档，直接从下面三种方式里选一种就行。

## 快速上手（3 选 1）

### 1) 在线直接用（推荐）

打开 [ChromaPrint3D](https://chromaprint3d.com/)。  
这个站点会和最新 Release 的功能保持同步，最适合第一次体验。

### 2) 下载桌面版（Win/macOS/Linux）

前往 Releases 页面下载对应平台安装包：  
<https://github.com/neroued/ChromaPrint3D/releases>

### 3) Docker 一键启动

```bash
docker run -d -p 8080:8080 --name chromaprint3d neroued/chromaprint3d:latest
```

启动后访问 `http://localhost:8080`。

> 镜像标签：`latest`（一体，开箱即用）、`api`（仅 API，分体部署）、`preview` / `preview-api`（预览版，发布前测试）。  
> 分体部署与预览版双轨部署详见 [docs/deployment.md](docs/deployment.md)。

## CLI 工具

| 命令 | 说明 |
|---|---|
| `raster_to_3mf` | 图像转多色 3MF 模型 |
| `gen_calibration_board` | 生成校准板 3MF |
| `build_colordb` | 从校准板照片构建 ColorDB |

> 矢量化 CLI 工具（raster_to_svg、evaluate_svg）已迁移至独立库 [neroued_vectorizer](https://github.com/neroued/neroued_vectorizer)。

## 开发者跳转

- 构建与打包：[docs/build.md](docs/build.md)
- 本地开发与调试：[docs/development.md](docs/development.md)
- 发布流程：[docs/development.md#8-发布流程](docs/development.md#8-发布流程)
- 部署：[docs/deployment.md](docs/deployment.md)
- 协作导航：[AGENTS.md](AGENTS.md)
- 模块索引：[docs/agents/README.md](docs/agents/README.md)
- 任务手册：[docs/agents/tasks/README.md](docs/agents/tasks/README.md)

---

<a id="english"></a>

## English

### Introduction

ChromaPrint3D turns images into multi-color 3MF models for 3D printing.  
It supports automatic Bambu Studio preset injection (0.2mm/0.4mm nozzle × face-up/face-down, 4 presets) and auto-matches model colors to the closest filament slot.  
You can also explicitly control base layers (`base_layers`), double-sided generation (`double_sided`), and transparent coating (`transparent_layer_mm`, FaceDown only) during conversion.
If you only want a quick hands-on try, pick one of the options below.

### Quick Start (Pick One)

#### 1) Use Web Directly (Recommended)

Open [ChromaPrint3D](https://chromaprint3d.com/).  
The hosted site tracks the latest Release feature set.

#### 2) Download Desktop App (Win/macOS/Linux)

Get the installer for your platform from Releases:  
<https://github.com/neroued/ChromaPrint3D/releases>

#### 3) One-command Docker Run

```bash
docker run -d -p 8080:8080 --name chromaprint3d neroued/chromaprint3d:latest
```

Then open `http://localhost:8080`.

> Image tags: `latest` (all-in-one), `api` (API only, split deployment), `preview` / `preview-api` (pre-release testing).  
> See [docs/deployment.md](docs/deployment.md) for split deployment and preview dual-track deployment details.

### CLI Tools

| Command | Description |
|---|---|
| `raster_to_3mf` | Convert image to multi-color 3MF model |
| `gen_calibration_board` | Generate calibration board 3MF |
| `build_colordb` | Build ColorDB from calibration photo |

> Vectorization CLI tools (raster_to_svg, evaluate_svg) have been migrated to the standalone library [neroued_vectorizer](https://github.com/neroued/neroued_vectorizer).

### Developer Entry

- Build and packaging: [docs/build.md](docs/build.md)
- Local development and debugging: [docs/development.md](docs/development.md)
- Release workflow: [docs/development.md#8-发布流程](docs/development.md#8-发布流程)
- Deployment: [docs/deployment.md](docs/deployment.md)
- Collaboration guide: [AGENTS.md](AGENTS.md)
- Module indexes: [docs/agents/README.md](docs/agents/README.md)
- Task playbooks: [docs/agents/tasks/README.md](docs/agents/tasks/README.md)
