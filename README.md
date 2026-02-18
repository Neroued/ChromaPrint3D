# ChromaPrint3D

[English](#english) | [中文](#中文)

---

<a id="中文"></a>

## 简介

ChromaPrint3D 是一个多色 3D 打印图像处理系统。它将输入图像的颜色映射为多通道打印配方，并导出可直接用于多色 3D 打印机的 3MF 文件。项目包含 C++20 核心库、命令行工具、HTTP 服务器、Web 前端以及基于 Beer-Lambert 定律的 Python 颜色建模管线。

### 主要功能

- **颜色数据库（ColorDB）**：加载/保存颜色数据库，支持 Lab 和 RGB 色彩空间的 KD-Tree 最近邻匹配
- **图像预处理**：缩放、去噪、透明度遮罩
- **配方匹配**：基于 ColorDB 的颜色匹配，可选基于物理模型的候选生成
- **3MF 导出**：通过 lib3mf 将体素网格转为 3MF 多色模型
- **校准板生成**：自动生成校准板 3MF 和元数据文件，支持 2-8 色
- **Web 服务**：完整的 HTTP API 和 Vue 3 前端界面
- **Python 建模管线**：基于薄层堆叠 Beer-Lambert 模型的颜色预测与参数拟合

## 目录结构

```
ChromaPrint3D/
├── core/               # C++ 核心库（颜色匹配、体素化、3MF 导出等）
├── apps/               # 命令行工具与 HTTP 服务器
├── web/                # Web 前端（Vue 3 + TypeScript + Naive UI）
├── modeling/           # Python 颜色建模管线
│   ├── core/           #   核心模块（前向模型、色彩空间、IO 工具）
│   └── pipeline/       #   管线步骤（5 步流程）
├── data/               # 运行时数据
│   ├── dbs/            #   ColorDB JSON 文件
│   ├── model_pack/     #   模型包 JSON 文件
│   └── recipes/        #   预计算配方文件（如 8 色校准板）
├── 3dparty/            # 第三方依赖（git submodules）
├── Dockerfile          # 运行时容器
└── Dockerfile.build    # 编译环境容器
```

## 获取代码

```bash
git clone https://github.com/neroued/ChromaPrint3D.git
cd ChromaPrint3D
git submodule update --init --recursive
```

## 构建

### 本地编译

#### 系统要求

- CMake >= 3.25
- C++20 编译器（推荐 g++-13 或更高版本）
- Node.js >= 22（用于 Web 前端构建）
- Python >= 3.10 + NumPy + OpenCV（用于建模管线，可选）

#### 编译 C++ 核心与应用

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

CMake 构建选项：

| 选项 | 默认值 | 说明 |
|---|---|---|
| `CHROMAPRINT3D_BUILD_APPS` | `ON` | 构建可执行文件 |
| `CHROMAPRINT3D_BUILD_TESTS` | `OFF` | 构建单元测试 |

编译完成后，可执行文件位于 `build/bin/`：

| 可执行文件 | 说明 |
|---|---|
| `chromaprint3d_server` | HTTP 服务器（含 Web API） |
| `gen_calibration_board` | 生成校准板 3MF 与元数据 |
| `build_colordb` | 从校准板照片构建 ColorDB |
| `image_to_3mf` | 将图像转为多色 3MF 模型 |
| `gen_stage` | 生成建模所需的阶梯色板 |
| `gen_representative_board` | 从配方文件生成代表性校准板 |

#### 构建 Web 前端

```bash
cd web
npm ci
npm run build
```

构建产物位于 `web/dist/`，可通过服务器 `--web` 参数指定该目录以提供静态文件服务。

### Docker 编译

项目提供 `Dockerfile.build` 作为一致的编译环境（Ubuntu 24.04 + g++-13 + CMake + Node.js 22）。

#### 1. 构建编译镜像

```bash
docker build -f Dockerfile.build -t chromaprint3d-build .
```

#### 2. 编译 C++

```bash
docker run --rm -v $(pwd):/src -w /src chromaprint3d-build \
  bash -c "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
```

#### 3. 构建 Web 前端

```bash
docker run --rm -v $(pwd):/src -w /src/web chromaprint3d-build \
  bash -c "npm ci && npm run build"
```

#### 4. 构建运行时镜像

完成上述编译后，可直接构建精简的运行时镜像：

```bash
docker build -t chromaprint3d .
```

启动容器：

```bash
docker run -d -p 8080:8080 --name chromaprint3d chromaprint3d
```

如需挂载自定义数据或模型包：

```bash
docker run -d -p 8080:8080 \
  -v ./my_data:/app/data \
  -v ./my_model_pack:/app/model_pack \
  --name chromaprint3d chromaprint3d
```

### 快速体验（使用预构建镜像）

如果不需要自行编译，可直接使用发布的 Docker 镜像：

```bash
docker run -d -p 8080:8080 --name chromaprint3d neroued/chromaprint3d:latest
```

访问 `http://localhost:8080` 即可使用。

## 使用

### 启动服务器

```bash
build/bin/chromaprint3d_server \
  --data data \
  --web web/dist \
  --model-pack data/model_pack/model_package.json \
  --port 8080
```

服务器参数：

| 参数 | 必需 | 默认值 | 说明 |
|---|---|---|---|
| `--data DIR` | 是 | — | 数据根目录（需包含 `dbs/` 和 `recipes/` 子目录） |
| `--port PORT` | 否 | 8080 | HTTP 端口 |
| `--host HOST` | 否 | 0.0.0.0 | 绑定地址 |
| `--web DIR` | 否 | — | 静态文件目录（Web 前端） |
| `--model-pack PATH` | 否 | — | 模型包 JSON 文件 |
| `--max-upload-mb N` | 否 | 50 | 最大上传文件大小（MB） |
| `--max-tasks N` | 否 | 4 | 最大并发任务数 |
| `--task-ttl N` | 否 | 3600 | 任务缓存过期时间（秒） |
| `--log-level LEVEL` | 否 | info | 日志级别 |

启动后访问 `http://localhost:8080` 即可使用 Web 界面。

### 命令行工具

```bash
# 生成 4 色校准板
build/bin/gen_calibration_board --channels 4 --out board.3mf --meta board.json

# 从校准板照片构建 ColorDB
build/bin/build_colordb --image calib_photo.png --meta board.json --out color_db.json

# 图像转 3MF
build/bin/image_to_3mf --image input.png --db color_db.json --out output.3mf --preview preview.png
```

### Python 建模管线

建模管线用于拟合颜色预测模型参数，需从项目根目录以模块方式运行：

```bash
# Step 1: 提取单色阶梯数据
python -m modeling.pipeline.step1_extract_stages --help

# Step 2: 拟合 Stage A 参数（E, k）
python -m modeling.pipeline.step2_fit_stage_a --help

# Step 3: 拟合 Stage B 参数（E, k, γ, δ, C₀）
python -m modeling.pipeline.step3_fit_stage_b --help

# Step 4: 选择代表性配方
python -m modeling.pipeline.step4_select_recipes --help

# Step 5: 构建运行时模型包
python -m modeling.pipeline.step5_build_model_package --help

# 生成 8 色校准板配方
python -m modeling.pipeline.gen_8color_board_recipes --help
```

## Web 界面

Web 前端提供三个功能页面：

1. **图像转换** — 上传图像，选择 ColorDB 和参数，转换为多色 3MF 模型
2. **校准工具（四色以下）** — 生成/下载校准板，上传照片构建 ColorDB
3. **八色校准（Beta）** — 八色校准板生成与下载，支持两张校准板的分步校准流程

开发模式下运行前端：

```bash
cd web
npm run dev
```

前端开发服务器默认在 `localhost:5173`，自动代理 `/api` 请求到 `localhost:8080`。

## 第三方依赖

所有 C++ 依赖通过 git submodules 管理，无需手动安装：

| 库 | 用途 |
|---|---|
| [OpenCV](https://github.com/opencv/opencv) | 图像处理（core, imgproc, imgcodecs） |
| [lib3mf](https://github.com/3MFConsortium/lib3mf) | 3MF 文件读写 |
| [spdlog](https://github.com/gabime/spdlog) | 结构化日志 |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON 序列化（header-only） |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | HTTP 服务器（header-only） |

OpenMP 为可选依赖，编译时自动检测，用于并行加速网格构建。

## 发布

项目使用 GitHub Actions 自动化发布。推送版本标签后自动触发：

```bash
git tag v1.0.0
git push origin v1.0.0
```

CI 会自动完成以下步骤：

1. 编译 C++ 和 Web 前端
2. 构建 Docker 镜像并推送至 [Docker Hub](https://hub.docker.com/r/neroued/chromaprint3d) 和 GitHub Container Registry
3. 创建 GitHub Release 并自动生成变更说明

**首次配置：** 需要在仓库 Settings → Secrets and variables → Actions 中添加：

| Secret | 说明 |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub 用户名 |
| `DOCKERHUB_TOKEN` | Docker Hub [Access Token](https://hub.docker.com/settings/security) |

GitHub Container Registry 使用仓库自带的 `GITHUB_TOKEN`，无需额外配置。

## 许可证

[Apache License 2.0](LICENSE)

---

<a id="english"></a>

## English

### Introduction

ChromaPrint3D is a multi-color 3D printing image processing system. It maps input image colors to multi-channel printing recipes and exports 3MF files ready for multi-color 3D printers. The project includes a C++20 core library, command-line tools, an HTTP server, a Vue 3 web frontend, and a Python color modeling pipeline based on Beer-Lambert's law.

### Key Features

- **Color Database (ColorDB)**: Load/save color databases with KD-Tree nearest-neighbor matching in Lab and RGB color spaces
- **Image Preprocessing**: Scaling, denoising, transparency masking
- **Recipe Matching**: ColorDB-based color matching with optional physics-model candidate generation
- **3MF Export**: Convert voxel grids to 3MF multi-color models via lib3mf
- **Calibration Board Generation**: Automatically generate calibration board 3MF and metadata files, supporting 2-8 colors
- **Web Service**: Full HTTP API and Vue 3 frontend interface
- **Python Modeling Pipeline**: Color prediction and parameter fitting based on thin-layer Beer-Lambert stacking model

### Directory Structure

```
ChromaPrint3D/
├── core/               # C++ core library (color matching, voxelization, 3MF export, etc.)
├── apps/               # Command-line tools and HTTP server
├── web/                # Web frontend (Vue 3 + TypeScript + Naive UI)
├── modeling/           # Python color modeling pipeline
│   ├── core/           #   Core modules (forward model, color space, IO utils)
│   └── pipeline/       #   Pipeline steps (5-step workflow)
├── data/               # Runtime data
│   ├── dbs/            #   ColorDB JSON files
│   ├── model_pack/     #   Model package JSON files
│   └── recipes/        #   Pre-computed recipe files (e.g., 8-color calibration)
├── 3dparty/            # Third-party dependencies (git submodules)
├── Dockerfile          # Runtime container
└── Dockerfile.build    # Build environment container
```

### Getting the Code

```bash
git clone https://github.com/neroued/ChromaPrint3D.git
cd ChromaPrint3D
git submodule update --init --recursive
```

### Building

#### Local Build

**Requirements:**

- CMake >= 3.25
- C++20 compiler (g++-13 or later recommended)
- Node.js >= 22 (for web frontend)
- Python >= 3.10 + NumPy + OpenCV (for modeling pipeline, optional)

**Build C++ core and applications:**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

CMake options:

| Option | Default | Description |
|---|---|---|
| `CHROMAPRINT3D_BUILD_APPS` | `ON` | Build executables |
| `CHROMAPRINT3D_BUILD_TESTS` | `OFF` | Build unit tests |

Executables are placed in `build/bin/`:

| Executable | Description |
|---|---|
| `chromaprint3d_server` | HTTP server with Web API |
| `gen_calibration_board` | Generate calibration board 3MF and metadata |
| `build_colordb` | Build ColorDB from calibration board photo |
| `image_to_3mf` | Convert image to multi-color 3MF model |
| `gen_stage` | Generate stage data for modeling pipeline |
| `gen_representative_board` | Generate representative board from recipe file |

**Build web frontend:**

```bash
cd web
npm ci
npm run build
```

Output is in `web/dist/`. Pass it to the server via the `--web` flag to serve static files.

#### Docker Build

The project provides `Dockerfile.build` as a consistent build environment (Ubuntu 24.04 + g++-13 + CMake + Node.js 22).

**1. Build the build image:**

```bash
docker build -f Dockerfile.build -t chromaprint3d-build .
```

**2. Compile C++:**

```bash
docker run --rm -v $(pwd):/src -w /src chromaprint3d-build \
  bash -c "cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j\$(nproc)"
```

**3. Build web frontend:**

```bash
docker run --rm -v $(pwd):/src -w /src/web chromaprint3d-build \
  bash -c "npm ci && npm run build"
```

**4. Build the runtime image:**

After completing the above steps:

```bash
docker build -t chromaprint3d .
```

Start the container:

```bash
docker run -d -p 8080:8080 --name chromaprint3d chromaprint3d
```

To mount custom data or model pack:

```bash
docker run -d -p 8080:8080 \
  -v ./my_data:/app/data \
  -v ./my_model_pack:/app/model_pack \
  --name chromaprint3d chromaprint3d
```

#### Quick Start (Pre-built Image)

If you don't need to build from source, use the pre-built Docker image:

```bash
docker run -d -p 8080:8080 --name chromaprint3d neroued/chromaprint3d:latest
```

Visit `http://localhost:8080` to start using.

### Usage

#### Starting the Server

```bash
build/bin/chromaprint3d_server \
  --data data \
  --web web/dist \
  --model-pack data/model_pack/model_package.json \
  --port 8080
```

Server options:

| Flag | Required | Default | Description |
|---|---|---|---|
| `--data DIR` | Yes | — | Data root directory (expects `dbs/` and `recipes/` inside) |
| `--port PORT` | No | 8080 | HTTP port |
| `--host HOST` | No | 0.0.0.0 | Bind address |
| `--web DIR` | No | — | Static files directory (web frontend) |
| `--model-pack PATH` | No | — | Model package JSON file |
| `--max-upload-mb N` | No | 50 | Max upload size in MB |
| `--max-tasks N` | No | 4 | Max concurrent tasks |
| `--task-ttl N` | No | 3600 | Task cache TTL in seconds |
| `--log-level LEVEL` | No | info | Log level |

Visit `http://localhost:8080` to use the web interface.

#### Command-Line Tools

```bash
# Generate a 4-color calibration board
build/bin/gen_calibration_board --channels 4 --out board.3mf --meta board.json

# Build ColorDB from a calibration board photo
build/bin/build_colordb --image calib_photo.png --meta board.json --out color_db.json

# Convert image to 3MF
build/bin/image_to_3mf --image input.png --db color_db.json --out output.3mf --preview preview.png
```

#### Python Modeling Pipeline

The modeling pipeline fits color prediction model parameters. Run from the project root as modules:

```bash
# Step 1: Extract single-color stage data
python -m modeling.pipeline.step1_extract_stages --help

# Step 2: Fit Stage A parameters (E, k)
python -m modeling.pipeline.step2_fit_stage_a --help

# Step 3: Fit Stage B parameters (E, k, γ, δ, C₀)
python -m modeling.pipeline.step3_fit_stage_b --help

# Step 4: Select representative recipes
python -m modeling.pipeline.step4_select_recipes --help

# Step 5: Build runtime model package
python -m modeling.pipeline.step5_build_model_package --help

# Generate 8-color calibration board recipes
python -m modeling.pipeline.gen_8color_board_recipes --help
```

### Web Interface

The web frontend has three tabs:

1. **Image Conversion** — Upload an image, select ColorDBs and parameters, convert to multi-color 3MF
2. **Calibration Tools (up to 4 colors)** — Generate/download calibration boards, upload photos to build ColorDB
3. **8-Color Calibration (Beta)** — 8-color calibration board generation, two-board step-by-step calibration workflow

For frontend development:

```bash
cd web
npm run dev
```

The dev server runs at `localhost:5173` and proxies `/api` requests to `localhost:8080`.

### Third-Party Dependencies

All C++ dependencies are managed via git submodules — no manual installation required:

| Library | Purpose |
|---|---|
| [OpenCV](https://github.com/opencv/opencv) | Image processing (core, imgproc, imgcodecs) |
| [lib3mf](https://github.com/3MFConsortium/lib3mf) | 3MF file read/write |
| [spdlog](https://github.com/gabime/spdlog) | Structured logging |
| [nlohmann/json](https://github.com/nlohmann/json) | JSON serialization (header-only) |
| [cpp-httplib](https://github.com/yhirose/cpp-httplib) | HTTP server (header-only) |

OpenMP is an optional dependency, auto-detected at build time for parallel mesh construction.

### Releasing

The project uses GitHub Actions for automated releases. Push a version tag to trigger:

```bash
git tag v1.0.0
git push origin v1.0.0
```

The CI pipeline will automatically:

1. Build C++ and web frontend
2. Build and push the Docker image to [Docker Hub](https://hub.docker.com/r/neroued/chromaprint3d) and GitHub Container Registry
3. Create a GitHub Release with auto-generated release notes

**First-time setup:** Add the following secrets in Settings → Secrets and variables → Actions:

| Secret | Description |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub [Access Token](https://hub.docker.com/settings/security) |

GitHub Container Registry uses the built-in `GITHUB_TOKEN` — no extra setup needed.

### License

[Apache License 2.0](LICENSE)
