# Web Frontend 模块索引

## 模块职责

`web/frontend/` 提供 Vue 3 + TypeScript 前端，负责参数输入、任务提交、轮询状态、结果展示和下载，支持 Browser/Electron 双运行时。

## 关键目录与入口

- 应用入口：`web/frontend/src/main.ts`、`web/frontend/src/App.vue`
- 页面组件：`web/frontend/src/components/`
- 参数域模型：`web/frontend/src/domain/params/`
- 业务编排：`web/frontend/src/services/`
- 组合式逻辑：`web/frontend/src/composables/`
- 全局状态：`web/frontend/src/stores/`
- API 客户端：`web/frontend/src/api/*.ts`（按业务域拆分）
- API 汇总导出：`web/frontend/src/api.ts`
- 运行时抽象：`web/frontend/src/runtime/`

## 常见改动落点

| 目标 | 入口文件 |
|---|---|
| 新增页面交互逻辑 | `src/components/*.vue` + `src/composables/` |
| 新增/调整后端参数映射 | `src/domain/params/convertParamBuilders.ts` + `src/api/convert.ts` |
| 调整任务轮询与状态处理 | `src/composables/useAsyncTask.ts` + `src/services/convertService.ts` |
| 调整分层预览行为 | `src/domain/result/layerPreview.ts` + `src/components/ResultPanel.vue` |
| 调整配方编辑器 | `src/components/recipeEditor/*.vue` + `src/api/recipeEditor.ts` + `src/composables/useRegionMap.ts` |
| 调整自定义配方与颜色预测 | `src/components/recipeEditor/CustomRecipeDialog.vue` + `src/api/recipeEditor.ts`（`predictRecipeColor`） |
| 调整 Browser/Electron 行为差异 | `src/runtime/*.ts` + `src/electron.d.ts` |

## 分层边界（强约束）

- `src/components/**/*.vue` 不应直接依赖 `src/api/*`。
- 页面组件优先调用 `src/services/*` 或 `src/composables/*`。
- 跨面板重复交互逻辑（上传、联动缩放、下载错误处理）优先沉淀到 `composables`。

## 前后端联动检查

- 后端参数改名或默认值变化时，检查：
  - `src/domain/params/convertDefaults.ts`
  - `src/domain/params/convertParamBuilders.ts`
  - `src/api/convert.ts`
  - `src/components/ParamPanel.vue`（新增/移除参数控件，含 `base_layers`、`double_sided` 等几何项）
- 上传约束变化时，检查：
  - `src/domain/upload/imageUploadValidation.ts`
  - `src/runtime/env.ts`

## 最小验证

```bash
cd web/frontend
npm run lint
npm run test
npm run build

# 需要仅排查类型问题时可单独执行
npm run typecheck
```

## 相关任务手册

- [docs/agents/tasks/add_api_endpoint.md](../../tasks/add_api_endpoint.md)
- [docs/agents/tasks/update_frontend_param.md](../../tasks/update_frontend_param.md)

