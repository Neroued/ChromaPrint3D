---
name: release-version
description: Guide the agent through ChromaPrint3D release workflows including preview and stable releases. Orchestrates version bump scripts, writes user-facing changelogs (version-manifest.json) in both Chinese and English, and creates release PRs. Use when the user mentions releasing, publishing, bumping version, creating a preview/rc release, writing changelogs, or generating version-manifest.
---

# ChromaPrint3D 发版流程

## 概述

发版通过仓库脚本自动完成版本号更新、manifest 生成和 PR 创建。Agent 的核心职责：

1. 确定版本号
2. 撰写面向用户的中英文 changelog
3. 执行对应脚本
4. 确认 PR 创建成功

## 前置条件

- 工作区干净（无未提交变更）
- 当前在 `master` 分支或可安全切换
- `gh` CLI 已认证

## 发版类型

### 预览版（Preview / RC）

```bash
./scripts/release-preview.sh <MAJOR.MINOR.PATCH> [rc_number] [changelog] [changelog_en]
```

- `rc_number` 可省略，脚本自动递增
- 分支命名：`preview/v<version>-rc.<N>`
- 合并后自动创建 tag 并触发 pre-release 流水线

### 正式版（Stable）

```bash
./scripts/release.sh <MAJOR.MINOR.PATCH> [changelog] [changelog_en]
```

- 分支命名：`release/v<version>`
- 会清除 `CHROMAPRINT3D_VERSION_PRERELEASE`
- 合并后自动创建 tag 并触发正式 release 流水线

## Changelog 撰写规范（关键）

Changelog 写入 `version-manifest.json`，展示给最终用户（Web 弹窗 + Electron 更新提示），**绝对不能出现开发相关内容**。

### 原则

- **面向用户**：用普通人能理解的语言，说明"用户能得到什么"
- **禁止出现**：模块名、函数名、文件名、架构概念（重构、迁移、子模块等）、任何技术细节
- **简洁有序**：每条 `- ` 开头，最重要的放最前面
- **中文为主**：`changelog` 用中文，`changelog_en` 用英文
- 纯粹的内部重构/代码质量改进如果对用户无感知影响，**不要写进 changelog**

### 改写示例

| commit message（内部） | changelog（面向用户） |
|---|---|
| feat: version update notification for web and Electron | 新增版本更新提醒，有新版本时自动通知 |
| refactor: migrate vectorization to submodule | 矢量化处理性能优化 |
| fix: restore calibration board meta JSON download | 修复校准板配置文件无法下载的问题 |
| feat: 自动聚类检测 + 三态聚类模式 | 智能颜色分析，自动识别图片最佳配色方案 |
| fix(backend): use WHOLE_ARCHIVE to preserve controller reg | 修复部分接口偶发不可用的问题 |
| chore: update CI matrix, bump deps | （不写入 changelog） |
| refactor: split ServerFacade into smaller classes | （不写入 changelog，除非有性能/稳定性改善） |

### 格式

JSON 中用 `\n` 分隔多行：

```
"- 新增版本更新提醒\n- 修复校准板配置下载问题"
```

## 执行流程

### 第 1 步：确定范围

```bash
# 查看上一个正式版 tag
git tag --sort=-v:refname -l 'v[0-9]*' | grep -v -- '-' | head -1

# 查看从上一个正式版到现在的提交
git log <上一个正式tag>..HEAD --oneline
```

如果是预览版，也从上一个正式 tag 开始统计，而非上一个 rc tag。

### 第 2 步：撰写 changelog

1. 阅读 commit 列表，识别用户可感知的变化
2. 按重要性排序，用面向用户的语言改写
3. 过滤掉纯内部改动（CI、重构、依赖更新等）
4. 同时撰写中文和英文版本
5. 将 changelog 呈现给用户确认后再执行脚本

### 第 3 步：执行脚本

确保工作区干净，然后执行：

```bash
# 预览版
./scripts/release-preview.sh 1.3.0 "" \
  "- 功能A\n- 修复B" \
  "- Feature A\n- Fix B"

# 正式版
./scripts/release.sh 1.3.0 \
  "- 功能A\n- 修复B" \
  "- Feature A\n- Fix B"
```

注意：预览版第二个参数（rc_number）传空字符串 `""` 让脚本自动递增。

### 第 4 步：确认结果

脚本成功后会输出 PR URL。提示用户：

1. 等待 CI 通过
2. Review 并合并 PR
3. 合并后 `release-tag` workflow 自动创建 tag
4. Tag 创建后触发 release pipeline 自动构建发布

## 注意事项

- 脚本会自动更新 `CMakeLists.txt`、`web/frontend/package.json`、`web/electron/package.json` 中的版本号
- 脚本会自动生成 `web/frontend/public/version-manifest.json`
- **不要**手动修改这些文件后再运行脚本
- 版本号必须是 `MAJOR.MINOR.PATCH` 格式，脚本会检查是否为上一版本的合法递增
- 如果工作区有未提交变更，脚本会拒绝执行——先 commit 或 stash
