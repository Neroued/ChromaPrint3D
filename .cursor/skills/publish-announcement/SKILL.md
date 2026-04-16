---
name: publish-announcement
description: Help the operator publish, list, and remove user-visible announcements on ChromaPrint3D. Use when the user mentions publishing an announcement, posting a notice, scheduling a maintenance banner, creating an announcement, or anything related to the `/api/v1/announcements` API and the `scripts/announce.sh` tool. This skill should NOT be used to deploy new versions — pair it with `release-version` when notifying users ahead of a scheduled upgrade.
---

# 发布公告（Announcement）工作流

## 概述

公告是用户打开 ChromaPrint3D 时顶部看到的 banner，**面向海内外用户同时展示**。
发布流程由后端 `/api/v1/announcements` 和 `scripts/announce.sh` 驱动：

1. 后端存储：`${data_dir}/announcements.json`（原子写，单文件）
2. 鉴权：所有写操作需要 `x-announcement-token` 头，且只能通过 HTTPS
3. 前端：启动时拉取一次；`/api/v1/health` 携带 `announcements_version`，
   该值变化时前端自动重新拉取
4. 用户侧：可选"不再提醒"（按 `id + updated_at` 维度持久化到 `localStorage`）

## Agent 职责

每次被触发时，Agent 必须依次完成：

1. 与用户确认目标：是**新发**、**修改**、**延期**、**撤回**，还是仅**查看**当前公告
2. 撰写中英双语内容，面向用户使用
3. 选择正确的 `type` / `severity` / `ends_at`（UTC，ISO8601）
4. 必要时提供 `scheduled_update_at`（用于前端倒计时）
5. 调用 `scripts/announce.sh`，**绝不把 token 明文写入聊天或 commit**
6. 核对服务端返回，必要时再次 `list` 验证

## 字段规范

| 字段 | 说明 |
|---|---|
| `id` | 幂等键，正则 `^[a-zA-Z0-9_-]{1,64}$`；同 `id` 再发会覆盖 |
| `type` | `info` / `warning` / `maintenance` |
| `severity` | `info` / `warning` / `error`；决定 banner 配色 |
| `title.zh` / `title.en` | 至少一种语言必填，≤ 2000 字符 |
| `body.zh` / `body.en` | 至少一种语言必填，≤ 2000 字符；支持多行，**禁止 HTML** |
| `starts_at` | 可选，ISO8601 UTC；省略表示立即生效 |
| `ends_at` | 必填，ISO8601 UTC，必须在未来且大于 `starts_at` |
| `scheduled_update_at` | 可选，升级窗口时间点，前端据此渲染倒计时 |
| `dismissible` | 默认 `true`；若需强制可见（如进行中的紧急维护）设为 `false` |

### 写作原则

- **面向用户**：告诉用户"会发生什么、对你有什么影响"
- **有明确时间锚点**：维护通知必须给出 `scheduled_update_at`
- **双语对齐**：中英文保持信息一致，不互相增减
- **别泄露内部语汇**：不提到模块名、commit、PR 编号
- **维护通知应提前 ≥ 1h**：方便跨时区用户看见

## 快速流程

### 1. 查看现有公告

```bash
./scripts/announce.sh list
```

`list` 不需要 token（GET 是公开的）。

### 2. 发布一条维护预告

先确保 token 已经通过环境变量导入（**不要写进命令行历史**）：

```bash
# 推荐做法：读文件，不回显
export CHROMAPRINT3D_ANNOUNCEMENT_TOKEN="$(cat ~/.chromaprint3d-announcement-token)"
# 或用专用文件变量（脚本会按需读取）：
export CHROMAPRINT3D_ANNOUNCEMENT_TOKEN_FILE=~/.chromaprint3d-announcement-token
```

再发布：

```bash
./scripts/announce.sh create \
    maintenance-2026-04-20 \
    maintenance \
    warning \
    '2026-04-20T10:30:00Z' \
    --title-zh '2026-04-20 维护通知' \
    --title-en 'Maintenance notice · 2026-04-20' \
    --body-zh '我们将在 UTC 10:00 进行约 10 分钟的升级，期间请保存您的工作。' \
    --body-en 'We will upgrade at 10:00 UTC for ~10 minutes. Please save your work.' \
    --starts-at '2026-04-19T10:00:00Z' \
    --scheduled-update-at '2026-04-20T10:00:00Z'
```

### 3. 用文件发布（推荐复杂文案）

```yaml
# announcements/maintenance-2026-04-20.yaml
id: maintenance-2026-04-20
type: maintenance
severity: warning
title_zh: '2026-04-20 维护通知'
title_en: 'Maintenance notice · 2026-04-20'
body_zh: |
  我们将在 UTC 10:00 进行约 10 分钟的升级。
  请在此之前保存您的工作，升级期间部分任务可能中断。
body_en: |
  We will upgrade at 10:00 UTC for ~10 minutes.
  Please save your work before then — in-flight tasks may be interrupted.
starts_at: '2026-04-19T10:00:00Z'
ends_at: '2026-04-20T10:30:00Z'
scheduled_update_at: '2026-04-20T10:00:00Z'
dismissible: true
```

```bash
./scripts/announce.sh from-file announcements/maintenance-2026-04-20.yaml
```

> 草稿 YAML 放在 `announcements/*.local.yaml`，已在 `.gitignore` 覆盖。

### 4. 延期 / 修改

同 `id` 重新 `create` 或 `from-file` 即可，服务端会保留 `created_at`、
刷新 `updated_at`，前端检测到 `announcements_version` 变化后重新拉取，
已"不再提醒"的用户在 `updated_at` 变动时会重新看到。

### 5. 撤回

```bash
./scripts/announce.sh delete maintenance-2026-04-20
```

撤回后前端在下一次 `health` 轮询（约 15 秒）内消失。

## 与发版（release-version）协作

若公告目的是"预告版本升级"，推荐顺序：

1. 本 skill 发公告，`scheduled_update_at` = 计划升级时间
2. 到点后使用 `release-version` skill 执行真正的发版
3. 升级完成后 `./scripts/announce.sh delete <id>` 撤回

## 安全自检（每次都做）

- [ ] 命令行里没有明文 token（用环境变量 / 文件）
- [ ] 不要在 PR / commit / chat 里贴 token
- [ ] 公告内容是**纯文本**，没有 HTML 或脚本
- [ ] 生产环境上传到的 URL 是 HTTPS
- [ ] 若操作失败返回 `404 not_found`：后端未配置 token，联系运维
- [ ] 若返回 `403 insecure_transport`：请求走了明文 HTTP，改走 HTTPS
- [ ] 若返回 `401 unauthorized`：token 错误 / 过期，联系运维轮换

## 参考

- 后端入口：`web/backend/src/presentation/announcement_auth_advice.cpp`
- 领域模型：`web/backend/src/application/announcement_service.{h,cpp}`
- 前端挂载：`web/frontend/src/components/AnnouncementBanner.vue`
- 运维文档：`docs/deployment.md`（token 部署章节）
- 契约：`docs/development.md`（API 契约章节）
