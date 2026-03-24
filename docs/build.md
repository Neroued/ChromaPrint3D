# ChromaPrint3D 构建指南

本文件是仓库的统一构建入口，集中说明：

- 本地构建（C++ / Web 前端）
- Docker 构建（`Dockerfile.build`）
- Electron 本地开发所需的后端前置构建（二进制准备）

如果你只想快速体验功能，请优先回到 [README.md](../README.md) 的“快速上手（3 选 1）”。

## 1. 获取源码

```bash
git clone https://github.com/neroued/ChromaPrint3D.git
cd ChromaPrint3D
git submodule update --init --recursive
```

## 2. 本地构建

### 2.1 环境要求

**必需：**

| 依赖 | 最低版本 | 说明 |
|---|---|---|
| CMake | 3.25 | 构建系统 |
| C++ 编译器 | C++20 | 推荐 `g++-13` 或更高 |
| OpenCV | 4.5+ | 图像处理依赖 |
| Potrace | — | 矢量化后端（neroued_vectorizer 依赖） |
| uuid 库 | — | UUID 生成（Drogon 依赖） |
| pkg-config | — | 库发现工具 |
| Node.js | 22+ | 前端构建 |

**推荐：**

| 依赖 | 说明 |
|---|---|
| Ninja | 更快的构建后端，CI 和 Docker 构建均使用 |
| ccache | 编译缓存，显著加速增量编译 |
| clang-format 18 | CI 强制检查 C++ 格式，建议本地安装以便提交前自检 |

**可选：**

| 依赖 | 说明 |
|---|---|
| zlib 1.2+ | 3MF ZIP Deflate 压缩后端（缺失时回退 Store） |
| Python 3.10+ | 建模管线 |

### 2.2 安装系统依赖（示例）

```bash
# Ubuntu / Debian
sudo apt install build-essential g++-13 cmake ninja-build pkg-config ccache \
    libopencv-dev libpotrace-dev uuid-dev zlib1g-dev

# macOS
brew install cmake ninja ccache opencv potrace ossp-uuid zlib
```

### 2.3 构建 C++ 核心与应用

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

CMake 选项：

| 选项 | 默认值 | 说明 |
|---|---|---|
| `CHROMAPRINT3D_BUILD_APPS` | `ON` | 构建可执行文件 |
| `CHROMAPRINT3D_BUILD_TESTS` | `OFF` | 构建单元测试 |
| `CHROMAPRINT3D_BUILD_INFER` | `ON` | 构建推理模块（ONNX Runtime 自动下载） |
| `CHROMAPRINT3D_3MF_ZLIB_REQUIRED` | `OFF` | 若为 `ON`，未找到 zlib 时配置失败 |
| `CHROMAPRINT3D_3MF_FORCE_STORE` | `OFF` | 强制 3MF ZIP 使用 store-only（禁用 Deflate） |

如需构建测试目标，可显式开启：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCHROMAPRINT3D_BUILD_TESTS=ON
cmake --build build -j$(nproc)
```

构建产物（`build/bin/`）：

| 可执行文件 | 说明 |
|---|---|
| `chromaprint3d_server` | HTTP 服务器（含 Web API） |
| `gen_calibration_board` | 生成校准板 3MF 与元数据 |
| `build_colordb` | 从校准板照片构建 ColorDB |
| `raster_to_3mf` | 图像转多色 3MF |
| `gen_stage` | 生成建模阶梯色板 |
| `gen_representative_board` | 从配方文件生成代表性校准板 |
| `svg_to_3mf` | SVG 转 3MF |
| `gen_test_preset_3mf` | 生成测试预设 3MF |

### 2.4 构建 Web 前端

```bash
cd web/frontend
npm ci
npm run build
```

构建产物位于 `web/frontend/dist/`，可通过服务端 `--web` 参数挂载。

如果你在中国大陆公开部署站点，可在构建前配置备案展示（浏览器模式生效，Electron 默认不展示）：

```bash
# 建议写入 web/frontend/.env.production.local（默认不会提交到 Git）
VITE_SITE_ICP_NUMBER=京ICP备12345678号-1
VITE_SITE_ICP_URL=https://beian.miit.gov.cn/
VITE_SITE_PUBLIC_SECURITY_RECORD_NUMBER=京公网安备11010502000001号
VITE_SITE_PUBLIC_SECURITY_RECORD_URL=https://beian.mps.gov.cn/
```

## 3. Docker 构建

项目提供 `Dockerfile.build` 作为一致构建环境（Ubuntu 22.04 + g++-13 + CMake + Node.js 22）。

### 3.1 构建编译镜像

```bash
docker build -f Dockerfile.build -t chromaprint3d-build .
```

### 3.2 编译 C++

```bash
docker run --rm -v $(pwd):/src -w /src chromaprint3d-build \
  bash -c "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
```

### 3.3 构建前端

```bash
docker run --rm -v $(pwd):/src -w /src/web/frontend chromaprint3d-build \
  bash -c "npm ci && npm run build"
```

### 3.4 构建运行时镜像

```bash
docker build -t chromaprint3d .
```

启动容器：

```bash
docker run -d -p 8080:8080 --name chromaprint3d chromaprint3d
```

挂载自定义数据/模型包：

```bash
docker run -d -p 8080:8080 \
  -v ./my_data:/app/data \
  -v ./my_model_pack:/app/model_pack \
  --name chromaprint3d chromaprint3d
```

## 4. Electron 后端预构建（仅构建）

如果你通过 `web/electron` 进行桌面壳开发，建议先构建后端二进制：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target chromaprint3d_server -j
```

Electron 开发启动命令与调试流程见 [docs/development.md](development.md)。  
本文件只维护构建相关步骤，避免与开发命令重复维护。

更多上下游说明见：

- [web/electron/README.md](../web/electron/README.md)
- [web/frontend/README.md](../web/frontend/README.md)
- [docs/development.md](development.md)

## 5. 最小排障

- 子模块为空目录：执行 `git submodule update --init --recursive`
- CMake 版本过低：升级到 3.25+
- C++20 报错：确认使用 `g++-13` 或更高
- 前端构建失败：确认 Node.js 版本满足 `22+`
- 链接报 `potrace` 符号缺失：安装 `libpotrace-dev`（Ubuntu）或 `potrace`（macOS）
- 链接报 `uuid` 符号缺失：安装 `uuid-dev`（Ubuntu）或 `ossp-uuid`（macOS）
- PR 格式检查失败：安装 `clang-format-18` 并在提交前执行 `clang-format -i <修改的文件>`

