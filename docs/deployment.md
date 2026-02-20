# ChromaPrint3D 跨域分体部署指南

本文档描述如何将 ChromaPrint3D 部署在两台服务器上：**云主机**（轻量门户）和**家庭主机**（高性能后端），以充分利用各自的资源优势。

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
    image: neroued/chromaprint3d:latest
    container_name: chromaprint3d
    restart: unless-stopped
    # 不暴露端口到宿主机，仅 backend 网络内部可达
    networks:
      - backend
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=256m
    # 跨域模式：限制 CORS 只允许云主机前端域名
    command:
      - "--data"
      - "/app/data"
      - "--web"
      - "/app/web"
      - "--model-pack"
      - "/app/model_pack/model_package.json"
      - "--port"
      - "8080"
      - "--cors-origin"
      - "https://chromaprint3d.com"
    deploy:
      resources:
        limits:
          cpus: "4.0"
          memory: 4G
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8080/api/health || exit 1"]
      interval: 30s
      timeout: 5s
      retries: 3
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "3"

  nginx:
    image: nginx:alpine
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

> **`--cors-origin https://chromaprint3d.com`** 是本次代码修改新增的参数。
> 启用后，服务端只接受来自该域名的跨域请求，且 session cookie 会自动使用 `SameSite=None; Secure`。

### 2.4 创建 Nginx 配置

```nginx
# ~/chromaprint3d-deploy/nginx/conf.d/api.conf

limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/s;
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
    add_header Strict-Transport-Security "max-age=63072000" always;

    # --- 连接/请求限制 ---
    limit_conn conn_limit 20;
    client_max_body_size 50m;

    # --- 普通 API ---
    location /api/ {
        limit_req zone=api_limit burst=20 nodelay;

        proxy_pass http://chromaprint3d:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 120s;
        proxy_connect_timeout 5s;
    }

    # --- 上传/转换接口：更严格限流 ---
    location /api/convert {
        limit_req zone=upload_limit burst=5 nodelay;

        proxy_pass http://chromaprint3d:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        client_max_body_size 50m;
    }

    location /api/calibration/build-colordb {
        limit_req zone=upload_limit burst=5 nodelay;

        proxy_pass http://chromaprint3d:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 300s;
        client_max_body_size 50m;
    }

    # --- 健康检查不对外暴露 ---
    location /api/health {
        allow 172.16.0.0/12;
        allow 10.0.0.0/8;
        allow 127.0.0.1;
        deny all;
    }

    # --- 只允许 /api 路径 ---
    location / {
        return 404;
    }
}
```

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
docker compose up -d

# 验证服务运行状态
docker compose ps
docker compose logs chromaprint3d --tail 20
```

## 第三步：构建前端

在开发机上修改前端环境变量并重新构建：

```bash
cd web

# 编辑 .env.production，设置 API 地址
# VITE_API_BASE=https://api.chromaprint3d.com:9443

# 构建（也可以通过命令行直接传入）
VITE_API_BASE=https://api.chromaprint3d.com:9443 npm run build
```

构建产物在 `web/dist/` 目录下。

> **CI/CD 构建：** 如果使用 GitHub Actions，在 release.yml 的前端构建步骤中设置环境变量：
> ```yaml
> - name: Build web frontend
>   working-directory: web
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
scp -r web/dist/* user@云主机IP:/var/www/chromaprint3d/
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
    # CSP 中明确允许前端连接家庭 API
    add_header Content-Security-Policy "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob: https://api.chromaprint3d.com:9443; connect-src 'self' https://api.chromaprint3d.com:9443;" always;

    root /var/www/chromaprint3d;
    index index.html;

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
| `web/src/api.ts` | `BASE` 从 `VITE_API_BASE` 环境变量读取；所有 `fetch` 调用和 URL 生成函数统一使用 `BASE` 前缀 |
| `web/.env.production` | 新增文件，`VITE_API_BASE` 环境变量模板 |

### 后端

| 文件 | 修改内容 |
|------|---------|
| `apps/server/server_options.h` | 新增 `--cors-origin` 命令行选项 |
| `apps/server/http_utils.h` | CORS 支持白名单模式：设置了 `--cors-origin` 时仅允许指定来源 |
| `apps/server/session.h` | 跨域模式下 session cookie 自动使用 `SameSite=None; Secure` |
| `apps/chromaprint3d_server.cpp` | 启动时读取并应用 `--cors-origin` 配置 |

**向后兼容：** 不传 `--cors-origin` 参数时，行为与修改前完全一致（允许所有来源、cookie 使用 `SameSite=Strict`）。单机 Docker 部署无需任何改动。

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

## 附录：单机部署（不分体）

如果不需要分体部署，直接使用 Docker 镜像即可，无需任何额外配置：

```bash
docker run -d -p 8080:8080 neroued/chromaprint3d:latest
```

此模式下 `VITE_API_BASE` 为空，前端和 API 同源，`--cors-origin` 不传，所有行为与修改前一致。
