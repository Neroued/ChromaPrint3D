# ChromaPrint3D

[![License](https://img.shields.io/github/license/neroued/ChromaPrint3D)](LICENSE)
[![Release](https://img.shields.io/github/v/release/neroued/ChromaPrint3D?include_prereleases)](https://github.com/neroued/ChromaPrint3D/releases)
[![Docker Pulls](https://img.shields.io/docker/pulls/neroued/chromaprint3d)](https://hub.docker.com/r/neroued/chromaprint3d)
[![Website](https://img.shields.io/website?url=https%3A%2F%2Fchromaprint3d.com)](https://chromaprint3d.com/)

[English](#english) | [中文](#中文)

---

<a id="中文"></a>

## 简介

ChromaPrint3D 可以把图片转换成多色 3D 打印可用的 3MF 模型。  
支持 Bambu Studio 预设参数自动写入（0.2mm/0.4mm 喷嘴 × 观赏面朝上/朝下，共 4 种预设），并自动将模型颜色匹配到最接近的耗材丝槽位。  
如果你只是想先体验效果，不需要先读一堆文档，直接从下面三种方式里选一种就行。

## 核心功能

- 图片 / SVG → 多色 3MF，4 种 Bambu Studio 预设自动注入
- 模型颜色自动匹配到耗材丝槽位（RGB 欧氏距离）
- 深度学习抠图（ONNX Runtime 推理）
- 矢量化处理：抗锯齿检测、边缘感知 SLIC 分割、逐像素标签细化
- 交互式配方编辑器：逐色重选、实时预览
- 校准板生成 + 拍照构建 ColorDB（支持 4/8 色板工作流）
- 高级几何控制：底板层数（`base_layers`）、双面打印（`double_sided`）、透明镀层（`transparent_layer_mm`，FaceDown 专属）
- 运维公告系统：双语 banner、HTTPS 写入、token 鉴权

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

> 完整可执行文件列表（含 `gen_stage`、`svg_to_3mf` 等）见 [docs/build.md](docs/build.md#23-构建-c-核心与应用)。

> 矢量化 CLI 工具（raster_to_svg、evaluate_svg）已迁移至独立库 [neroued_vectorizer](https://github.com/neroued/neroued_vectorizer)。

## 开发者跳转

- 构建与打包：[docs/build.md](docs/build.md)
- 本地开发与调试：[docs/development.md](docs/development.md)
- 发布流程：[docs/development.md#8-发布流程](docs/development.md#8-发布流程)
- 部署：[docs/deployment.md](docs/deployment.md)
- 运维公告（升级预告/维护通知）：[docs/deployment.md#公告系统announcements](docs/deployment.md#公告系统announcements)
- 协作导航：[AGENTS.md](AGENTS.md)
- 模块索引：[docs/agents/README.md](docs/agents/README.md)
- 任务手册：[docs/agents/tasks/README.md](docs/agents/tasks/README.md)

## 致谢

ChromaPrint3D 构建在众多优秀开源项目之上，特别感谢：

- C++ 核心 / 算法：[OpenCV](https://opencv.org)、[spdlog](https://github.com/gabime/spdlog)、[Clipper2](https://github.com/AngusJohnson/Clipper2)、[lunasvg](https://github.com/sammycage/lunasvg)、[nlohmann/json](https://github.com/nlohmann/json)、[earcut.hpp](https://github.com/mapbox/earcut.hpp)、[ONNX Runtime](https://onnxruntime.ai)、[jemalloc](https://jemalloc.net)、[Little CMS](https://www.littlecms.com)
- Web 后端：[Drogon](https://github.com/drogonframework/drogon)
- Web 前端：[Vue](https://vuejs.org)、[Naive UI](https://www.naiveui.com)、[Pinia](https://pinia.vuejs.org)、[Vue I18n](https://vue-i18n.intlify.dev)、[Chart.js](https://www.chartjs.org)、[Vite](https://vitejs.dev)
- 桌面壳：[Electron](https://www.electronjs.org)
- Python 建模：[NumPy](https://numpy.org)、[SciPy](https://scipy.org)、[OpenCV](https://opencv.org)、[msgpack](https://msgpack.org)、[matplotlib](https://matplotlib.org)
- 运行时系统依赖：[Potrace](https://potracer.com/)（通过 [neroued_vectorizer](https://github.com/neroued/neroued_vectorizer)）

## 许可证

本项目采用 Apache License 2.0，完整文本见 [LICENSE](LICENSE)。

## 贡献

欢迎通过 Issue / Pull Request 参与。提交前请阅读 [AGENTS.md](AGENTS.md) 与 [Git 分支与 PR 规范](.cursor/rules/git-branch-pr-policy.mdc)。

---

<a id="english"></a>

## English

### Introduction

ChromaPrint3D turns images into multi-color 3MF models for 3D printing.  
It supports automatic Bambu Studio preset injection (0.2mm/0.4mm nozzle × face-up/face-down, 4 presets) and auto-matches model colors to the closest filament slot.  
If you only want a quick hands-on try, pick one of the options below.

### Features

- Image / SVG to multi-color 3MF with auto-injected Bambu Studio presets (4 variants)
- Auto-match model colors to filament slots (RGB Euclidean distance)
- Deep-learning matting (ONNX Runtime inference)
- Vectorization: anti-aliasing detection, edge-aware SLIC, per-pixel label refinement
- Interactive recipe editor: per-color replacement with live preview
- Calibration board generation + photo-based ColorDB build (4/8-color workflows)
- Advanced geometry controls: base layers (`base_layers`), double-sided printing (`double_sided`), transparent coating (`transparent_layer_mm`, FaceDown only)
- Announcement system: bilingual banner, HTTPS-only writes, token-gated mutations

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

> Full list of executables (including `gen_stage`, `svg_to_3mf`, etc.) is in [docs/build.md](docs/build.md#23-构建-c-核心与应用).

> Vectorization CLI tools (raster_to_svg, evaluate_svg) have been migrated to the standalone library [neroued_vectorizer](https://github.com/neroued/neroued_vectorizer).

### Developer Entry

- Build and packaging: [docs/build.md](docs/build.md)
- Local development and debugging: [docs/development.md](docs/development.md)
- Release workflow: [docs/development.md#8-发布流程](docs/development.md#8-发布流程)
- Deployment: [docs/deployment.md](docs/deployment.md)
- Operations announcements (upgrade notices / maintenance banners): [docs/deployment.md#公告系统announcements](docs/deployment.md#公告系统announcements)
- Collaboration guide: [AGENTS.md](AGENTS.md)
- Module indexes: [docs/agents/README.md](docs/agents/README.md)
- Task playbooks: [docs/agents/tasks/README.md](docs/agents/tasks/README.md)

### Acknowledgments

ChromaPrint3D builds on many excellent open-source projects. Special thanks to:

- C++ core / algorithms: [OpenCV](https://opencv.org), [spdlog](https://github.com/gabime/spdlog), [Clipper2](https://github.com/AngusJohnson/Clipper2), [lunasvg](https://github.com/sammycage/lunasvg), [nlohmann/json](https://github.com/nlohmann/json), [earcut.hpp](https://github.com/mapbox/earcut.hpp), [ONNX Runtime](https://onnxruntime.ai), [jemalloc](https://jemalloc.net), [Little CMS](https://www.littlecms.com)
- Web backend: [Drogon](https://github.com/drogonframework/drogon)
- Web frontend: [Vue](https://vuejs.org), [Naive UI](https://www.naiveui.com), [Pinia](https://pinia.vuejs.org), [Vue I18n](https://vue-i18n.intlify.dev), [Chart.js](https://www.chartjs.org), [Vite](https://vitejs.dev)
- Desktop shell: [Electron](https://www.electronjs.org)
- Python modeling: [NumPy](https://numpy.org), [SciPy](https://scipy.org), [OpenCV](https://opencv.org), [msgpack](https://msgpack.org), [matplotlib](https://matplotlib.org)
- Runtime system dependency: [Potrace](https://potracer.com/) (via [neroued_vectorizer](https://github.com/neroued/neroued_vectorizer))

### License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for the full text.

### Contributing

Issues and pull requests are welcome. Please read [AGENTS.md](AGENTS.md) and the [Git branch & PR policy](.cursor/rules/git-branch-pr-policy.mdc) before submitting.
