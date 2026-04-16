# ChromaPrint3D AGENTS 协作指南

本文件是仓库级导航入口，目标是让人类开发者与 AI agent 都能在最少跳转内定位模块、约束和常见任务落点。

## 1 分钟上手

1. 先读总览：[README.md](README.md)
2. 再按任务类型跳转模块索引：[docs/agents/](docs/agents/)
3. 改代码前确认规则：`.cursor/rules/`
4. 改代码后执行对应验证与文档同步检查

## 项目全景（任务导向）

- `core/`：C++ 核心算法与导出链路（图像处理、配方匹配、体素/网格、3MF）
- `apps/`：CLI 工具入口（图像转 3MF、校准板、ColorDB 构建等）
- `web/backend/`：Drogon HTTP API（`/api/v1/*`）
- `web/frontend/`：Vue3 + TypeScript 前端
- `modeling/`：Python 建模与拟合流水线（Stage A/B、配方选择、模型包）
- `docs/`：开发、部署与专题文档

## 按任务快速定位

| 任务 | 首先查看 |
|---|---|
| 新增/修改 API 端点 | [docs/agents/web/backend/README.md](docs/agents/web/backend/README.md) |
| 新增/修改服务启动参数 | [docs/agents/web/backend/README.md](docs/agents/web/backend/README.md) |
| 调整图像到 3MF 转换行为 | [docs/agents/core/README.md](docs/agents/core/README.md) |
| 调整配方匹配策略 | [docs/agents/core/README.md](docs/agents/core/README.md) + [docs/agents/tasks/tune_color_match.md](docs/agents/tasks/tune_color_match.md) |
| 新增/修改命令行工具参数 | [docs/agents/apps/README.md](docs/agents/apps/README.md) + [docs/agents/tasks/extend_cli_flag.md](docs/agents/tasks/extend_cli_flag.md) |
| 修改前端页面与参数面板 | [docs/agents/web/frontend/README.md](docs/agents/web/frontend/README.md) |
| 矢量化质量评估与回归检测 | [neroued_vectorizer](https://github.com/neroued/neroued_vectorizer)（已迁移） |
| 更新建模拟合流程 | [docs/agents/modeling/README.md](docs/agents/modeling/README.md) |
| 发布/撤回运维公告 | [docs/agents/tasks/publish_announcement.md](docs/agents/tasks/publish_announcement.md) + [.cursor/skills/publish-announcement/SKILL.md](.cursor/skills/publish-announcement/SKILL.md) |

## 模块索引入口

- [docs/agents/core/README.md](docs/agents/core/README.md)
- [docs/agents/apps/README.md](docs/agents/apps/README.md)
- [docs/agents/web/backend/README.md](docs/agents/web/backend/README.md)
- [docs/agents/web/frontend/README.md](docs/agents/web/frontend/README.md)
- [docs/agents/modeling/README.md](docs/agents/modeling/README.md)
- 任务手册：[docs/agents/tasks/](docs/agents/tasks/)

## Hooks（自动化执行）

以下行为由 `.cursor/hooks.json` 配置的 hooks 自动执行，无需手动操作：

- C++ 文件编辑后自动运行 `clang-format`（afterFileEdit hook）
- 前端文件编辑后自动运行 `prettier`（afterFileEdit hook）
- 前端文件编辑后自动检测 i18n 硬编码 UI 文本（afterFileEdit prompt hook）
- 危险 Shell 命令拦截：`rm -rf /`、`force push master` 等（beforeShellExecution hook）
- 敏感文件保护：`.env`、私钥、凭据文件等（beforeReadFile hook）
- Agent 结束时检查文档是否同步（stop hook）

## 全局协作规则（必须遵守）

- C++ 编码规范（格式化、跨平台、代码结构与工具函数复用）：[.cursor/rules/cpp-coding-standards.mdc](.cursor/rules/cpp-coding-standards.mdc)
- 涉及 API/参数/构建/用户行为变化时同步文档：[.cursor/rules/sync-docs.mdc](.cursor/rules/sync-docs.mdc)
- 前后端联动大型修改必须执行端到端闭环检查：[.cursor/rules/end-to-end-change-checklist.mdc](.cursor/rules/end-to-end-change-checklist.mdc)
- 前端分层边界与组件职责约束：[.cursor/rules/frontend-architecture.mdc](.cursor/rules/frontend-architecture.mdc)
- 前端 i18n 国际化规范（禁止硬编码文案、翻译 key 同步、locale 格式化）：[.cursor/rules/frontend-i18n.mdc](.cursor/rules/frontend-i18n.mdc)
- Git 提交流程强制规范（禁止直提 `master`，必须分支 + PR）：[.cursor/rules/git-branch-pr-policy.mdc](.cursor/rules/git-branch-pr-policy.mdc)

## 标准工作流（建议）

1. **确认范围**：只改与目标相关的模块与文件
2. **先索引后实现**：先读对应模块索引，再改代码
3. **验证优先**：先跑最小可验证命令，再做更完整检查
4. **同步文档**：行为变化必须同步 [README.md](README.md) / [docs/](docs/)

## 提交前检查清单

- C++ 格式化已由 hook 自动执行，无需手动检查
- 前端格式化已由 hook 自动执行；提交前可运行 `cd web/frontend && npm run format:check && npm run lint && npm run test && npm run build` 做全量验证
- 是否完成最小验证（构建、测试或功能性检查）
- 是否更新受影响的文档入口（[README.md](README.md)、[docs/development.md](docs/development.md)、[docs/deployment.md](docs/deployment.md)、[docs/agents/](docs/agents/)）
- 是否避免引入重复工具函数或可复用逻辑分叉
- 是否避免平台相关硬编码（路径、系统 API、脚本依赖）

## 文档维护约定

- [README.md](README.md) 放全局能力与使用入口，不堆实现细节
- [docs/development.md](docs/development.md) 放开发与 API 行为契约
- [docs/deployment.md](docs/deployment.md) 放部署拓扑、环境与安全参数
- [docs/agents/](docs/agents/) 只做“导航与落点”，不复制原文档大段内容
- 高频任务优先沉淀到 [docs/agents/tasks/](docs/agents/tasks/)

## 文档更新触发矩阵

| 变更类型 | 必须同步更新 |
|---|---|
| API 路由、请求/响应字段、错误码变化 | [README.md](README.md)、[docs/development.md](docs/development.md)、[docs/agents/web/backend/README.md](docs/agents/web/backend/README.md)、相关任务手册 |
| CLI 参数或默认值变化 | [README.md](README.md)、[docs/development.md](docs/development.md)、[docs/agents/apps/README.md](docs/agents/apps/README.md)、[docs/agents/tasks/extend_cli_flag.md](docs/agents/tasks/extend_cli_flag.md) |
| 构建/部署流程变化 | [README.md](README.md)、[docs/development.md](docs/development.md)、[docs/deployment.md](docs/deployment.md)、[docs/agents/tasks/sync_docs_after_behavior_change.md](docs/agents/tasks/sync_docs_after_behavior_change.md) |
| 用户可见前端行为变化 | [README.md](README.md)、[docs/development.md](docs/development.md)、[docs/agents/web/frontend/README.md](docs/agents/web/frontend/README.md) |
| 模块入口或职责边界变化 | [docs/agents/README.md](docs/agents/README.md) 与对应模块索引 |

## PR 检查要求

每次提交都应使用仓库 PR 模板中的协作清单，确保覆盖：

- 代码验证
- 文档同步
- AGENTS 索引/任务手册更新

