# 发布运维公告

## 适用场景

- 计划维护 / 版本升级前需要提前向用户展示 banner
- 紧急状态（降级、临时变更）需要对海内外在线用户同时宣告
- 撤回已发布的公告

本手册只覆盖"写入与撤回"流程，不覆盖实际代码部署；部署流程见 [release-version skill](../../../.cursor/skills/release-version/SKILL.md) 与 [docs/deployment.md](../../deployment.md)。

## 影响文件（按优先级）

- `scripts/announce.sh` — CLI 工具（list / create / delete / from-file）
- `.cursor/skills/publish-announcement/SKILL.md` — Agent 协作 SOP
- `docs/deployment.md#公告系统announcements` — 部署契约与 token 管理
- 运行时数据：`${data_dir}/announcements.json`（后端原子写入，无需手改）

## 前置条件

1. 后端启动参数带 `--announcement-token <TOKEN>`（或等效 env：`CHROMAPRINT3D_ANNOUNCEMENT_TOKEN`）
2. 生产环境走 HTTPS，反代透传 `X-Forwarded-Proto: https`
3. 运维机 IP 在 Nginx `location /api/v1/announcements` 的 `allow` 白名单内
4. 本地：
   ```bash
   export CHROMAPRINT3D_API_URL=https://api.chromaprint3d.com:9443
   export CHROMAPRINT3D_ANNOUNCEMENT_TOKEN_FILE=~/.chromaprint3d-announcement-token
   ```

## 实施步骤

1. 拉取当前公告，避免重复：
   ```bash
   ./scripts/announce.sh list
   ```
2. 起草中英双语标题/正文（面向用户视角）。
3. 选定：
   - `type`（`info` / `warning` / `maintenance`）
   - `severity`（`info` / `warning` / `error`，决定 banner 配色）
   - `ends_at`（UTC，未来时间）
   - 可选：`starts_at`、`scheduled_update_at`、`dismissible`
4. 执行创建：
   ```bash
   ./scripts/announce.sh create <id> <type> <severity> <ends_at> \
     --title-zh '...' --title-en '...' \
     --body-zh  '...' --body-en  '...' \
     --scheduled-update-at '...'   # 维护通知时强烈建议
   ```
5. 再次 `list` 核对落盘结果。
6. 公告失效或升级完成后撤回：
   ```bash
   ./scripts/announce.sh delete <id>
   ```

## 风险点

- **token 泄露**：绝不把 token 明文粘到 PR / commit / chat，使用 `CHROMAPRINT3D_ANNOUNCEMENT_TOKEN_FILE`。
- **无 token 配置**：后端对 POST/DELETE 会返回 `404 not_found`，不是 `401`；不要误判为"路由挂了"。
- **明文 HTTP**：生产写入请求走 HTTP 会返回 `403 insecure_transport`，本地联调可加 `--announcement-allow-http`。
- **body 超限**：单条请求 ≤ 16 KiB；触发时返回 `413 payload_too_large`。
- **用户已 dismiss 旧版本**：修改内容时 `updated_at` 会自动刷新，前端 dismiss 按 `{id, updated_at}` 存储，会自动再次展示。
- **跨时区用户**：维护通知至少提前 1 小时发布，`scheduled_update_at` 必填。

## 回归检查

- 发布后冒烟：
  ```bash
  curl -s "$CHROMAPRINT3D_API_URL/api/v1/announcements" | jq
  curl -s "$CHROMAPRINT3D_API_URL/api/v1/health"          | jq '.data.announcements_version, .data.active_announcement_count'
  ```
- 前端浏览器端在新标签打开正式站点，确认 banner 出现且可 dismiss。
- 撤回公告后，确认 `announcements` 列表为空且 `active_announcement_count` 为 0。
- 文档检查：
  - 是否更新 [README.md](../../../README.md) / [docs/development.md](../../development.md) / [docs/deployment.md](../../deployment.md)（仅在行为/契约变化时）
  - 是否更新 [.cursor/skills/publish-announcement/SKILL.md](../../../.cursor/skills/publish-announcement/SKILL.md)
