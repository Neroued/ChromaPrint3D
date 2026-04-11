# Modeling 模块索引

## 模块职责

`modeling/` 提供 Python 建模流水线，用于从实拍/标注数据拟合前向模型参数，并产出 C++ 运行时可用的模型包与代表性配方。

## 关键目录与入口

- 核心模块：`modeling/core/`
  - 前向模型：`forward_model.py`
  - 色彩空间：`color_space.py`
  - 优化器：`adam.py`
  - IO 与标签规范化：`io_utils.py`
- 流水线：`modeling/pipeline/`
  - `step1_extract_stages.py`
  - `step2_fit_stage_a.py`
  - `step3_fit_stage_b.py`（支持 `--vendor`/`--material-type`/`--material` 参数，输出 JSON 含 `scope`+`channel_keys` 元数据）
  - `step4_select_recipes.py`
  - `step5_build_model_package.py`（输出 `.msgpack` 二进制格式，含 `scope` 元数据）
- 工具：`modeling/tools/`
  - `dump_model_package.py`（msgpack → 可读 JSON，调试用）
- 评估：`modeling/eval/`

## 常见改动落点

| 目标 | 入口文件 |
|---|---|
| 调整色彩空间转换 | `modeling/core/color_space.py` |
| 调整前向模型与预测 | `modeling/core/forward_model.py` |
| 调整 Stage A 拟合策略 | `modeling/pipeline/step2_fit_stage_a.py` |
| 调整 Stage B 损失/正则 | `modeling/pipeline/step3_fit_stage_b.py` |
| 调整代表性配方选取 | `modeling/pipeline/step4_select_recipes.py` |
| 调整模型包内容与阈值 | `modeling/pipeline/step5_build_model_package.py` |
| 检查/调试 msgpack 模型包 | `modeling/tools/dump_model_package.py` |

## 数据与产物约定

- 输入 ColorDB：`modeling/dbs/*.json`
- 关键输出目录：
  - `modeling/output/params/`（Stage B 参数 JSON，含 `param_type`/`scope`/`channel_keys`，可复制到 `data/models/forward/` 供 C++ 运行时加载）
  - `modeling/output/recipes/`
  - `modeling/output/packages/`（`.msgpack` 模型包）
  - `modeling/output/reports/`
- 运行时前向模型数据：`data/models/forward/*.json`（服务端启动时由 `DataRepository` 加载到 `ForwardModelRegistry`）
- 依赖：`msgpack`（Python 包，step5 和 dump 工具需要）

## 最小验证

从仓库根目录执行：

```bash
python -m modeling.pipeline.step2_fit_stage_a --help
python -m modeling.pipeline.step3_fit_stage_b --help
python -m modeling.pipeline.step5_build_model_package --help
```

如果变更会影响 C++ 运行时模型门控参数（threshold/margin），同步检查：

- [README.md](../../../README.md) 中模型参数说明
- `web/backend` 参数解析与默认值
- `web/frontend` 参数构建与默认值

## 相关任务手册

- [docs/agents/tasks/tune_color_match.md](../tasks/tune_color_match.md)
- [docs/agents/tasks/update_frontend_param.md](../tasks/update_frontend_param.md)

