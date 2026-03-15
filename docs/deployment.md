# ChromaPrint3D 跨域分体部署指南

本文档描述如何将 ChromaPrint3D 部署在两台服务器上：**云主机**（轻量门户）和**家庭主机**（高性能后端），以充分利用各自的资源优势。

## AGENTS 协作入口

如需让人和 agent 在部署相关改动中保持一致，请先参考：

- [AGENTS.md](../AGENTS.md)
- [docs/agents/README.md](agents/README.md)
- [docs/agents/tasks/sync_docs_after_behavior_change.md](agents/tasks/sync_docs_after_behavior_change.md)

## 架构概览

```
                        ┌─────────────────────────────┐
                        │         用户浏览器            │
                        └─────┬───────────────┬───────┘
                              │               │
               静态页面 (HTTPS 443)    API 请求 (HTTPS 9443)
                              │               │
                              ▼               ▼
                   ┌──────────────┐   ┌───────────────────┐
                   │   云主机      │   │    家庭主机         │
                   │  Nginx       │   │  Nginx (:9443)    │
                   │  静态文件     │   │      ↓            │
                   │              │   │  ChromaPrint3D    │
                   │              │   │  Docker (:8080)   │
                   └──────────────┘   └───────────────────┘
                   chromaprint3d.com         api.chromaprint3d.com
```

**为什么这样部署：**

- 云主机有备案域名和 80/443 端口，但性能弱、带宽小 → 只托管静态文件（几 MB）
- 家庭主机性能强、带宽大，但 80/443/8080 被封 → 用 9443 端口提供 API 服务
- 图片上传和 3MF 下载直连家庭主机，不经过云主机带宽瓶颈

## 前提条件

- 域名已完成 ICP 备案，A 记录指向云主机
- 家庭主机有公网 IP（固定或动态）
- 家庭主机 9443 端口未被 ISP 封禁

## 第一步：DNS 配置

在域名服务商控制台添加解析记录：

| 类型 | 主机记录 | 记录值 | 说明 |
|------|---------|--------|------|
| A | @ | `<云主机IP>` | 已有，指向云主机 |
| A | api | `<家庭公网IP>` | **新增**，指向家庭主机 |

如果家庭 IP 是动态的，参考本文末尾的 [DDNS 自动更新](#附录-ddns-自动更新) 章节。

## 第二步：家庭主机部署

### 2.1 申请 TLS 证书

由于家庭主机 80 端口不可用，使用 DNS-01 验证方式申请证书。推荐 [acme.sh](https://github.com/acmesh-official/acme.sh)：

```bash
# 安装 acme.sh
curl https://get.acme.sh | sh

# 设置 DNS API 密钥（以阿里云为例，其他服务商参考 acme.sh 文档）
export Ali_Key="你的AccessKey"
export Ali_Secret="你的AccessSecret"

# 申请证书
acme.sh --issue --dns dns_ali -d api.chromaprint3d.com

# 安装到指定目录
mkdir -p /etc/nginx/ssl
acme.sh --install-cert -d api.chromaprint3d.com \
  --key-file       /etc/nginx/ssl/api.chromaprint3d.com.key \
  --fullchain-file /etc/nginx/ssl/api.chromaprint3d.com.crt \
  --reloadcmd      "docker exec home-nginx nginx -s reload"
```

> 不同 DNS 服务商使用不同插件：阿里云 `dns_ali`、Cloudflare `dns_cf`、DNSPod `dns_dp`。
> 完整列表：https://github.com/acmesh-official/acme.sh/wiki/dnsapi
>
> acme.sh 会自动创建 crontab 定时续期。

### 2.2 创建部署目录

```bash
mkdir -p ~/chromaprint3d-deploy/nginx/conf.d
cd ~/chromaprint3d-deploy
```

### 2.3 创建 docker-compose.yml

```yaml
# ~/chromaprint3d-deploy/docker-compose.yml
version: "3.8"

services:
  chromaprint3d:
    # API-only 镜像（不含前端），跨域部署专用
    # 也可用 neroued/chromaprint3d:vX.Y.Z-api 固定版本
    image: neroued/chromaprint3d:api
    container_name: chromaprint3d
    restart: unless-stopped
    # 不暴露端口到宿主机，仅 backend 网络内部可达
    networks:
      - backend
    user: "10001:10001"
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=256m,uid=10001,gid=10001
      # 任务结果 3MF 落盘目录（read_only 下需单独挂载可写 tmpfs）
      - /app/data/tmp:noexec,nosuid,size=512m,uid=10001,gid=10001
    environment:
      # 限制每个 OpenMP 并行区域的线程数，防止多个 worker 同时运行时超额订阅
      # 公式：OMP_NUM_THREADS ≈ 可用核心数 / max_tasks
      # 示例：4 核 / 4 worker = 1；16 核 / 6 worker ≈ 3
      - OMP_NUM_THREADS=1
    # 跨域模式：限制 CORS 只允许云主机前端域名
    command:
      - "--data"
      - "/app/data"
      - "--model-pack"
      - "/app/model_pack/model_package.json"
      - "--port"
      - "8080"
      - "--cors-origin"
      - "https://chromaprint3d.com"
      - "--require-cors-origin"
      - "1"
      # 公网基线建议：优先收紧上传/队列/缓存占用（按机器规格再调优）
      - "--max-upload-mb"
      - "30"
      - "--max-queue"
      - "128"
      - "--max-owner-tasks"
      - "4"
      - "--task-ttl"
      - "900"
      - "--max-result-mb"
      - "64"       # 3MF 已落盘，内存中仅存 preview/mask；可按 max_tasks × 0.5MB 估算
      # 可选：进一步收紧像素上限
      # - "--max-pixels"
      # - "12000000"
    # docker compose 模式下可生效的资源限制（不要仅写 deploy.resources）
    cpus: "4.0"
    mem_limit: 4g
    memswap_limit: 4g
    pids_limit: 512
    healthcheck:
      test: ["CMD-SHELL", "bash -c 'echo > /dev/tcp/localhost/8080'"]
      interval: 10s
      timeout: 3s
      retries: 3
      start_period: 5s
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "3"

  nginx:
    image: nginx:1.27.5-alpine@sha256:<替换为实际digest>
    container_name: home-nginx
    restart: unless-stopped
    ports:
      - "9443:443"
    volumes:
      - ./nginx/conf.d:/etc/nginx/conf.d:ro
      - /etc/nginx/ssl:/etc/nginx/ssl:ro
    networks:
      - backend
      - default
    depends_on:
      chromaprint3d:
        condition: service_healthy

networks:
  backend:
    internal: true    # chromaprint3d 容器不可访问外网
```

> 如需固定版本，可使用 `neroued/chromaprint3d:vX.Y.Z-api` 或 digest 引用：
> ```bash
> docker pull neroued/chromaprint3d:api
> docker image inspect neroued/chromaprint3d:api --format '{{index .RepoDigests 0}}'
>
> docker pull nginx:1.27.5-alpine
> docker image inspect nginx:1.27.5-alpine --format '{{index .RepoDigests 0}}'
> ```
>
> 资源限制生效验证（Compose 非 Swarm）：
> ```bash
> docker compose config | rg "cpus|mem_limit|memswap_limit|pids_limit"
> docker inspect chromaprint3d --format 'NanoCPUs={{.HostConfig.NanoCpus}} Memory={{.HostConfig.Memory}} PidsLimit={{.HostConfig.PidsLimit}}'
> ```

> 建议生产启用 `--require-cors-origin 1`。未配置 `--cors-origin` 时服务将启动失败（fail-closed），避免误开放跨域来源。
>
> `--cors-origin https://chromaprint3d.com` 启用后，服务端只接受来自该域名的跨域请求，且 session cookie 会自动使用 `SameSite=None; Secure`。

### 2.4 创建 Nginx 配置

```nginx
# ~/chromaprint3d-deploy/nginx/conf.d/api.conf

limit_req_zone $binary_remote_addr zone=api_limit:10m rate=20r/s;
limit_req_zone $binary_remote_addr zone=upload_limit:10m rate=2r/s;
limit_conn_zone $binary_remote_addr zone=conn_limit:10m;

server {
    listen 443 ssl http2;
    server_name api.chromaprint3d.com;

    # --- TLS ---
    ssl_certificate     /etc/nginx/ssl/api.chromaprint3d.com.crt;
    ssl_certificate_key /etc/nginx/ssl/api.chromaprint3d.com.key;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers on;
    ssl_session_cache   shared:SSL:10m;

    # --- 安全响应头 ---
    server_tokens off;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;

    # --- 压缩（JSON/文本类响应） ---
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_min_length 1024;
    gzip_types application/json text/plain text/css application/javascript image/svg+xml;

    # --- 连接/请求限制 ---
    limit_conn conn_limit 20;
    client_max_body_size 50m;

    # --- 健康检查：不参与业务限流，避免误报 503 ---
    location = /api/v1/health {
        proxy_pass http://chromaprint3d:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 10s;
    }

    # --- 校准板生成：计算量大，需要长超时 ---
    location /api/v1/calibration/boards {
        limit_req zone=upload_limit burst=5 nodelay;

        proxy_pass http://chromaprint3d:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        proxy_connect_timeout 5s;
    }

    # --- 上传接口：严格限流 + 长超时 ---
    location /api/v1/calibration/colordb {
        limit_req zone=upload_limit burst=5 nodelay;

        proxy_pass http://chromaprint3d:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        client_max_body_size 50m;
    }

    location /api/v1/convert/ {
        limit_req zone=upload_limit burst=5 nodelay;

        proxy_pass http://chromaprint3d:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        client_max_body_size 50m;
    }

    location = /api/v1/matting/tasks {
        limit_req zone=upload_limit burst=5 nodelay;

        proxy_pass http://chromaprint3d:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 30s;
        client_max_body_size 50m;
    }

    location /api/v1/matting/ {
        limit_req zone=api_limit burst=20 nodelay;

        proxy_pass http://chromaprint3d:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 30s;
    }

    location /api/v1/session/databases/upload {
        limit_req zone=upload_limit burst=5 nodelay;

        proxy_pass http://chromaprint3d:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        client_max_body_size 50m;
    }

    # --- 制品下载：限制单连接速率，防止大文件下载占满上行带宽 ---
    location ~ ^/api/v1/tasks/.+/artifacts/ {
        limit_req zone=api_limit burst=10 nodelay;
        limit_rate 5m;       # 单连接限速 5MB/s（约 40Mbps）

        proxy_pass http://chromaprint3d:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        proxy_buffering off;         # 流式传输，不在 Nginx 缓冲大文件
    }

    # --- 普通 API（兜底） ---
    location /api/v1/ {
        limit_req zone=api_limit burst=40 nodelay;

        proxy_pass http://chromaprint3d:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_connect_timeout 5s;
    }

    # --- 只允许 /api/v1 路径 ---
    location / {
        return 404;
    }
}
```

> 阈值协同建议（避免“网关先拒绝”或“后端再拒绝”不一致）：
> - Nginx 的 `client_max_body_size` 应 **不小于** 后端 `--max-upload-mb`（默认 50MB）。
> - 如果调整了后端上传上限，请同步修改该配置中的全局与各上传路由 `client_max_body_size`。
> - 后端 `--max-pixels`（默认 `16777216`）限制的是**解码后像素数**，与文件大小限制独立；仅放宽 Nginx 限制并不能避免后端 `413 image_too_large`。
> - Web 部署若要“限制更紧”，请同时下调后端 `--max-upload-mb`/`--max-pixels` 与 Nginx `client_max_body_size`。
> - `--max-result-mb` 现在仅追踪内存中的制品大小（preview、mask、layer previews）。3MF 结果已自动落盘到 `data_dir/tmp/results/`，不计入内存预算。因此该值可以比旧版本大幅降低（如 64MB），同时容纳更多已完成任务。
> - 失败或中断的 3MF 导出会主动回收未提交的临时文件；服务启动时也会清扫 `data_dir/tmp/results/` 与 fallback 结果目录中的历史残留，避免旧的半写入文件持续占满 `tmpfs`。
> - 如需排查结果目录占用，可执行 `docker exec chromaprint3d du -sh /app/data/tmp/results` 查看总量，再用 `docker exec chromaprint3d ls -lhS /app/data/tmp/results | head -30` 定位最大文件。

### 2.5 配置防火墙

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp       # SSH
sudo ufw allow 9443/tcp     # ChromaPrint3D API
sudo ufw enable
```

### 2.6 启动服务

```bash
cd ~/chromaprint3d-deploy
docker compose pull
docker compose up -d --remove-orphans

# 验证服务运行状态
docker compose ps
docker compose logs chromaprint3d --tail 20
```

## 第三步：构建前端

在开发机上配置前端环境变量并重新构建（推荐使用本地覆盖文件，避免提交敏感/站点专属信息）：

```bash
cd web/frontend

# 建议新建 .env.production.local（该文件默认不进 Git）
cat > .env.production.local <<'EOF'
VITE_API_BASE=https://api.chromaprint3d.com:9443
VITE_SITE_ICP_NUMBER=京ICP备12345678号-1
VITE_SITE_ICP_URL=https://beian.miit.gov.cn/
VITE_SITE_PUBLIC_SECURITY_RECORD_NUMBER=京公网安备11010502000001号
VITE_SITE_PUBLIC_SECURITY_RECORD_URL=https://beian.mps.gov.cn/
EOF

# 构建
npm run build
```

构建产物在 `web/frontend/dist/` 目录下。

> `VITE_SITE_ICP_NUMBER` 与 `VITE_SITE_PUBLIC_SECURITY_RECORD_NUMBER` 仅在浏览器模式页脚展示，Electron 模式默认不展示。

> **CI/CD 构建：** 如果使用 GitHub Actions，在 release.yml 的前端构建步骤中设置环境变量：
> ```yaml
> - name: Build web frontend
>   working-directory: web/frontend
>   env:
>     VITE_API_BASE: https://api.chromaprint3d.com:9443
>   run: |
>     npm ci
>     npm run build
> ```

## 第四步：云主机部署

### 4.1 申请 TLS 证书

云主机有 80 端口，直接用 HTTP-01 验证：

```bash
sudo apt install certbot
sudo certbot certonly --standalone -d chromaprint3d.com
```

certbot 会自动配置定时续期。

### 4.2 上传前端文件

```bash
# 在云主机上创建目录
ssh user@云主机IP "sudo mkdir -p /var/www/chromaprint3d"

# 从开发机上传构建产物
scp -r web/frontend/dist/* user@云主机IP:/var/www/chromaprint3d/
```

### 4.3 安装并配置 Nginx

```bash
sudo apt install nginx
```

创建 `/etc/nginx/sites-available/chromaprint3d`：

```nginx
server {
    listen 80;
    server_name chromaprint3d.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name chromaprint3d.com;

    ssl_certificate     /etc/letsencrypt/live/chromaprint3d.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/chromaprint3d.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    server_tokens off;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "geolocation=(), microphone=(), camera=()" always;
    # CSP 中明确允许前端连接家庭 API
    add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https://api.chromaprint3d.com:9443; connect-src 'self' https://api.chromaprint3d.com:9443;" always;

    # --- 压缩（HTML/CSS/JS/SVG/JSON） ---
    gzip on;
    gzip_vary on;
    gzip_proxied any;
    gzip_min_length 1024;
    gzip_types text/plain text/css application/javascript application/json image/svg+xml;

    root /var/www/chromaprint3d;
    index index.html;

    # index.html 禁止长期缓存，确保每次发布可及时拿到新资源索引
    location = /index.html {
        add_header Cache-Control "no-store, must-revalidate";
        try_files $uri =404;
    }

    # SPA 路由回退
    location / {
        try_files $uri $uri/ /index.html;
    }

    # Vite 构建产物带 hash，可长期缓存
    location /assets/ {
        expires 30d;
        add_header Cache-Control "public, immutable";
    }
}
```

启用配置：

```bash
sudo ln -s /etc/nginx/sites-available/chromaprint3d /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

### 4.4 配置防火墙

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp
sudo ufw allow 80/tcp      # HTTP → HTTPS 重定向
sudo ufw allow 443/tcp     # HTTPS
sudo ufw enable
```

## 可选：一键自动化部署脚本

仓库内提供了 `scripts/deploy_split.sh`，用于把本文的高频操作串成一次执行：

1. 本地构建 `web` 前端
2. 将 `web/frontend/dist` 上传到云主机，**原子切换**前端目录并保留回滚快照
3. 在家庭主机执行后端更新流程：`docker compose pull && docker compose up -d --remove-orphans`，并等待容器就绪

### 使用方式

```bash
cd /path/to/ChromaPrint3D

# 1) 创建本地配置（包含服务器地址等敏感信息，不建议提交）
cp scripts/deploy_split.env scripts/deploy_split.local.env

# 2) 编辑配置
vim scripts/deploy_split.local.env

# 3) 执行一键部署
./scripts/deploy_split.sh scripts/deploy_split.local.env
```

也可以传入自定义配置路径：

```bash
./scripts/deploy_split.sh /path/to/your-deploy.env
```

### 配置说明（核心项）

- `CLOUD_HOST`：云主机 SSH 地址（例如 `user@1.2.3.4`）
- `CLOUD_WEB_ROOT`：云主机前端静态目录（例如 `/var/www/chromaprint3d`）
- `HOME_HOST`：家庭主机 SSH 地址
- `HOME_DEPLOY_DIR`：家庭主机 `docker-compose.yml` 所在目录（例如 `~/chromaprint3d-deploy`）
- `VITE_API_BASE`：前端构建时注入的 API 地址（例如 `https://api.chromaprint3d.com:9443`）
- `CLOUD_KEEP_ROLLBACKS`：云主机前端回滚快照保留数量（默认 `2`）
- `HOME_HEALTH_TIMEOUT_SECONDS`：后端更新后等待容器就绪超时（默认 `120` 秒）
- `HOME_ROLLBACK_DIR`：家庭主机保存回滚快照（compose 配置与镜像引用）的目录

> 建议提前配置 SSH 免密登录；若目标目录需要提权，保留 `CLOUD_USE_SUDO=1` 或按需启用 `HOME_USE_SUDO=1`。

## 第五步：验证

1. 访问 `https://chromaprint3d.com`，应看到 ChromaPrint3D 前端页面
2. 打开浏览器开发者工具 → Network，确认 API 请求发向 `https://api.chromaprint3d.com:9443`
3. 测试上传图片、转换、下载 3MF 文件等功能
4. 确认 session cookie 跨域正常工作（Application → Cookies 中应出现 `session` cookie）

## 代码修改说明

本部署方案涉及以下代码修改（均已完成）：

### 前端

| 文件 | 修改内容 |
|------|---------|
| `web/frontend/src/api.ts` | `BASE` 从 `VITE_API_BASE` 环境变量读取；所有 `fetch` 调用和 URL 生成函数统一使用 `BASE` 前缀 |
| `web/frontend/.env.production` | 新增文件，`VITE_API_BASE` 环境变量模板 |

### 后端

| 文件 | 修改内容 |
|------|---------|
| `web/backend/server_main.cpp` | Drogon 启动入口，统一读取并应用服务配置 |
| `web/backend/src/config/server_config.*` | 扩展命令行参数（含 `--cors-origin`、任务/资源上限） |
| `web/backend/src/presentation/api_v1_controller.*` | 统一 CORS 白名单与 cookie 策略（跨域自动 `SameSite=None; Secure`） |
| `web/backend/src/infrastructure/session_store.*` | 会话与 token 管理（CSPRNG token） |

**兼容说明：** 不传 `--cors-origin` 参数时仍可同源部署（允许来源更宽松，cookie 默认 `SameSite=Strict`）；跨域部署建议显式设置 `--cors-origin`。

---

## 附录：DDNS 自动更新

如果家庭公网 IP 是动态的，需要定时更新 `api` A 记录。

### 使用 acme.sh + 阿里云 CLI

```bash
#!/bin/bash
# /opt/ddns/update.sh
CURRENT_IP=$(curl -s https://api.ipify.org)
LAST_IP=$(cat /opt/ddns/last_ip 2>/dev/null)

if [ "$CURRENT_IP" != "$LAST_IP" ]; then
    aliyun alidns UpdateDomainRecord \
        --RecordId "你的记录ID" \
        --RR "api" \
        --Type "A" \
        --Value "$CURRENT_IP"
    echo "$CURRENT_IP" > /opt/ddns/last_ip
    echo "$(date): IP changed to $CURRENT_IP" >> /opt/ddns/ddns.log
fi
```

```bash
chmod +x /opt/ddns/update.sh

# 每 5 分钟执行一次
crontab -e
# 添加: */5 * * * * /opt/ddns/update.sh
```

## 资源调优指南

默认参数以低配公网部署为基准（4 核 / 4GB）。部署在性能更强的机器上时，
按以下原则调整。

### 核心参数与硬件的关系

| 参数 | 计算方式 | 4c/4GB 基线 | 16c/32GB 示例 |
|------|----------|-------------|---------------|
| `--max-tasks` | CPU 核心 / OMP 线程数 | 4 | 6 |
| `OMP_NUM_THREADS` | CPU 核心 / max_tasks | 1 | 3 |
| `--http-threads` | 2–8，通常 4 足够 | 4 | 4 |
| `--max-result-mb` | 3MF 已落盘，仅 preview/mask 在内存 | 64 | 64 |
| `--max-upload-mb` | 按可接受的最大图片大小 | 30 | 100 |
| `--max-pixels` | 按内存余量，每像素峰值约 200B | 12M | 33M |
| `--max-owner-tasks` | 单用户不应独占全部 worker | 4 | 8 |
| `--max-queue` | 控制排队深度 | 128 | 128 |
| `--task-ttl` | 结果保留时间（秒） | 900 | 1800 |
| Docker cpus | 留 1 核给 OS + Nginx | 4.0 | 15.0 |
| Docker mem_limit | 留 4GB 给 OS | 4g | 28g |

### 内存预估

单个 pipeline 任务峰值内存约 3–4GB（大图 + 体素网格 + Mesh 构建 + 流式压缩）。
总峰值 ≈ max_tasks × 4GB + 基础开销 (~1GB)。
确保 Docker mem_limit > 该总峰值，并为 OS page cache 留出余量。

### OMP_NUM_THREADS 的重要性

pipeline 内部的体素填充（geo.cpp）、Mesh 构建（export_3mf.cpp）、
梯度扩散（gradient.cpp）、抖动（dither.cpp）均使用 OpenMP 并行。
不设置此变量时，每个并行区域默认使用容器全部可见核心，
多 worker 同时运行会严重超额订阅，导致 CPU 争用和性能下降。

### Nginx 下载限速

制品下载路由建议配置 `limit_rate`，防止并发大文件下载占满上行带宽：
- 100Mbps 上行：`limit_rate 5m`（约 40Mbps/连接，2 连接可并行）
- 50Mbps 上行：`limit_rate 3m`
- 同时建议 `proxy_buffering off` 以支持流式传输

## 附录：单机部署（不分体）

如果不需要分体部署，直接使用 all-in-one 镜像即可，前端与后端同源运行：

```bash
docker run -d -p 8080:8080 neroued/chromaprint3d:latest
```

访问 `http://localhost:8080` 即可使用，无需配置 `--cors-origin`。

### Docker 镜像标签说明

| 标签 | 内容 | 用途 |
|------|------|------|
| `latest` / `vX.Y.Z` / `vX.Y` | 前端 + 后端一体 | 普通用户开箱即用 |
| `api` / `vX.Y.Z-api` / `vX.Y-api` | 仅后端 API | 跨域分体部署，前端由 CDN/Nginx 独立托管 |
