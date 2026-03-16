# Apps 模块索引

## 模块职责

`apps/` 提供面向用户和开发调试的 CLI 程序入口，链接 `ChromaPrint3D::core`。

## 可执行程序与源码映射

由 `apps/CMakeLists.txt` 定义：

| 可执行文件 | 源文件 |
|---|---|
| `gen_calibration_board` | `apps/gen_calibration_board.cpp` |
| `build_colordb` | `apps/build_colordb.cpp` |
| `raster_to_3mf` | `apps/raster_to_3mf.cpp` |
| `svg_to_3mf` | `apps/svg_to_3mf.cpp` |
| `gen_representative_board` | `apps/gen_representative_board.cpp` |
| `gen_stage` | `apps/gen_stage.cpp` |
| `gen_test_preset_3mf` | `apps/gen_test_preset_3mf.cpp` |

## 常见改动落点

| 目标 | 入口文件 |
|---|---|
| 新增/修改 CLI 参数 | 对应 `apps/*.cpp` 的参数解析段 |
| 调整默认值 | 对应 `apps/*.cpp` 默认配置处 |
| 输出文件命名规则调整 | 对应 `apps/*.cpp` 输出路径生成逻辑 |
| 将核心新能力暴露到 CLI | 对应 `apps/*.cpp` + `core/include/chromaprint3d/*.h` |

> 矢量化 CLI 工具（raster_to_svg、evaluate_svg）已迁移至独立库 [neroued_vectorizer](https://github.com/neroued/neroued_vectorizer)。

## 变更注意事项

- CLI 参数语义变化后，同步更新：
  - [README.md](../../../README.md)
  - [docs/development.md](../../development.md)
  - 必要时更新对应任务手册
- 不在多个 CLI 重复写相同参数解析工具，优先复用已有实现。

## 最小验证

```bash
cmake --build build -j$(nproc)
build/bin/raster_to_3mf --help
```

## 相关任务手册

- [docs/agents/tasks/extend_cli_flag.md](../tasks/extend_cli_flag.md)
- [docs/agents/tasks/add_api_endpoint.md](../tasks/add_api_endpoint.md)（当 CLI 行为需与 API 参数保持一致时）

