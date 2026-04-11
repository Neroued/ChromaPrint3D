#!/usr/bin/env bash
#
# Deploy split architecture:
#   1) Build web frontend locally and upload to cloud host
#   2) Atomic web root switch on cloud host with rollback snapshot
#   3) Update backend stack on home host (pull -> up -d --remove-orphans)
#
# Usage:
#   ./scripts/deploy_split.sh [config-file]
#
# Config bootstrap:
#   cp scripts/deploy_split.env scripts/deploy_split.local.env
#   # edit scripts/deploy_split.local.env
#   ./scripts/deploy_split.sh scripts/deploy_split.local.env
#
set -euo pipefail

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }
is_positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_CONFIG="$ROOT_DIR/scripts/deploy_split.env"
CONFIG_PATH="${1:-$DEFAULT_CONFIG}"

[[ -f "$CONFIG_PATH" ]] || die "Config file not found: $CONFIG_PATH (copy scripts/deploy_split.env to scripts/deploy_split.local.env first)"

# shellcheck disable=SC1090
source "$CONFIG_PATH"

required_vars=(
    CLOUD_HOST
    CLOUD_WEB_ROOT
    HOME_HOST
    HOME_DEPLOY_DIR
)
for var_name in "${required_vars[@]}"; do
    [[ -n "${!var_name:-}" ]] || die "Missing required config: $var_name"
done

CLOUD_SSH_PORT="${CLOUD_SSH_PORT:-22}"
HOME_SSH_PORT="${HOME_SSH_PORT:-22}"
CLOUD_UPLOAD_TAR="${CLOUD_UPLOAD_TAR:-/tmp/chromaprint3d-web.tar.gz}"
CLOUD_USE_SUDO="${CLOUD_USE_SUDO:-1}"
HOME_USE_SUDO="${HOME_USE_SUDO:-0}"
CLOUD_RELOAD_NGINX="${CLOUD_RELOAD_NGINX:-1}"
WEB_INSTALL_DEPS="${WEB_INSTALL_DEPS:-0}"
CLOUD_KEEP_ROLLBACKS="${CLOUD_KEEP_ROLLBACKS:-2}"
HOME_HEALTH_TIMEOUT_SECONDS="${HOME_HEALTH_TIMEOUT_SECONDS:-120}"
HOME_ROLLBACK_DIR="${HOME_ROLLBACK_DIR:-$HOME/.chromaprint3d-rollbacks}"

is_positive_integer "$CLOUD_KEEP_ROLLBACKS" || die "CLOUD_KEEP_ROLLBACKS must be a positive integer"
is_positive_integer "$HOME_HEALTH_TIMEOUT_SECONDS" || die "HOME_HEALTH_TIMEOUT_SECONDS must be a positive integer"

for cmd in ssh scp tar npm; do
    command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
done

cloud_ssh_opts=(-p "$CLOUD_SSH_PORT")
home_ssh_opts=(-p "$HOME_SSH_PORT")

# Remote sudo may require an interactive TTY.
if [[ "$CLOUD_USE_SUDO" == "1" ]]; then
    cloud_ssh_opts+=(-tt)
fi
if [[ "$HOME_USE_SUDO" == "1" ]]; then
    home_ssh_opts+=(-tt)
fi

if [[ "$WEB_INSTALL_DEPS" == "1" ]]; then
    info "Installing web dependencies with npm ci..."
    (cd "$ROOT_DIR/web/frontend" && npm ci)
fi

info "Building web frontend..."
BUILD_ENV=()
if [[ -n "${VITE_API_BASE:-}" ]]; then
    info "Using VITE_API_BASE=$VITE_API_BASE"
    BUILD_ENV+=(VITE_API_BASE="$VITE_API_BASE")
fi
if [[ -n "${VITE_UPDATE_MANIFEST_URL:-}" ]]; then
    info "Using VITE_UPDATE_MANIFEST_URL=$VITE_UPDATE_MANIFEST_URL"
    BUILD_ENV+=(VITE_UPDATE_MANIFEST_URL="$VITE_UPDATE_MANIFEST_URL")
fi
if [[ -n "${VITE_STABLE_URL:-}" ]]; then
    info "Using VITE_STABLE_URL=$VITE_STABLE_URL"
    BUILD_ENV+=(VITE_STABLE_URL="$VITE_STABLE_URL")
fi
if [[ -n "${VITE_PREVIEW_URL:-}" ]]; then
    info "Using VITE_PREVIEW_URL=$VITE_PREVIEW_URL"
    BUILD_ENV+=(VITE_PREVIEW_URL="$VITE_PREVIEW_URL")
fi
if [[ -n "${VITE_UMAMI_HOST:-}" ]]; then
    info "Using VITE_UMAMI_HOST=$VITE_UMAMI_HOST"
    BUILD_ENV+=(VITE_UMAMI_HOST="$VITE_UMAMI_HOST")
fi
if [[ -n "${VITE_UMAMI_WEBSITE_ID:-}" ]]; then
    info "Using VITE_UMAMI_WEBSITE_ID=$VITE_UMAMI_WEBSITE_ID"
    BUILD_ENV+=(VITE_UMAMI_WEBSITE_ID="$VITE_UMAMI_WEBSITE_ID")
fi
if (( ${#BUILD_ENV[@]} > 0 )); then
    (cd "$ROOT_DIR/web/frontend" && env "${BUILD_ENV[@]}" npm run build)
else
    (cd "$ROOT_DIR/web/frontend" && npm run build)
fi

WEB_DIST_DIR="$ROOT_DIR/web/frontend/dist"
[[ -d "$WEB_DIST_DIR" ]] || die "Build output not found: $WEB_DIST_DIR"

TMP_TAR="$(mktemp /tmp/chromaprint3d-web.XXXXXX.tar.gz)"
trap 'rm -f "$TMP_TAR"' EXIT
tar -C "$WEB_DIST_DIR" -czf "$TMP_TAR" .
info "Packaged web assets into $TMP_TAR"

info "Uploading web bundle to cloud host..."
scp -P "$CLOUD_SSH_PORT" "$TMP_TAR" "$CLOUD_HOST:$CLOUD_UPLOAD_TAR"

info "Applying web bundle on cloud host..."
CLOUD_SCRIPT="$(mktemp /tmp/chromaprint3d-cloud-deploy.XXXXXX.sh)"
cat > "$CLOUD_SCRIPT" <<'CLOUD_SCRIPT_BODY'
#!/usr/bin/env bash
set -euo pipefail

cloud_web_root="$1"
cloud_upload_tar="$2"
cloud_use_sudo="$3"
cloud_reload_nginx="$4"
cloud_keep_rollbacks="$5"

if [[ -z "$cloud_web_root" || "$cloud_web_root" == "/" ]]; then
    echo "ERROR: Refusing to deploy to unsafe CLOUD_WEB_ROOT: '$cloud_web_root'" >&2
    exit 1
fi
if ! [[ "$cloud_keep_rollbacks" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: CLOUD_KEEP_ROLLBACKS must be a positive integer." >&2
    exit 1
fi

run_cmd() {
    if [[ "$cloud_use_sudo" == "1" ]]; then
        sudo "$@"
    else
        "$@"
    fi
}

if [[ "$cloud_use_sudo" == "1" ]]; then
    sudo -v
fi

web_parent="$(dirname "$cloud_web_root")"
web_name="$(basename "$cloud_web_root")"
deploy_stamp="$(date +%Y%m%d%H%M%S)"
staging_dir="$web_parent/.${web_name}.staging.${deploy_stamp}"
rollback_dir="$web_parent/.${web_name}.rollback.${deploy_stamp}"

run_cmd mkdir -p "$web_parent"
run_cmd rm -rf "$staging_dir"
run_cmd mkdir -p "$staging_dir"
run_cmd tar -xzf "$cloud_upload_tar" -C "$staging_dir"
rm -f "$cloud_upload_tar"

if ! run_cmd test -f "$staging_dir/index.html"; then
    run_cmd rm -rf "$staging_dir"
    echo "ERROR: Uploaded web bundle does not contain index.html." >&2
    exit 1
fi

had_previous=0
if run_cmd test -d "$cloud_web_root"; then
    had_previous=1
    run_cmd mv "$cloud_web_root" "$rollback_dir"
fi

if ! run_cmd mv "$staging_dir" "$cloud_web_root"; then
    run_cmd rm -rf "$staging_dir"
    if [[ "$had_previous" == "1" ]] && run_cmd test -d "$rollback_dir"; then
        run_cmd mv "$rollback_dir" "$cloud_web_root"
    fi
    echo "ERROR: Failed to switch web root atomically." >&2
    exit 1
fi

reload_ok=1
if [[ "$cloud_reload_nginx" == "1" ]]; then
    if [[ "$cloud_use_sudo" == "1" ]]; then
        if ! sudo nginx -t; then
            reload_ok=0
        elif ! sudo systemctl reload nginx; then
            reload_ok=0
        fi
    else
        if ! nginx -t; then
            reload_ok=0
        elif ! systemctl reload nginx; then
            reload_ok=0
        fi
    fi
fi

if [[ "$reload_ok" != "1" ]]; then
    run_cmd rm -rf "$cloud_web_root"
    if [[ "$had_previous" == "1" ]] && run_cmd test -d "$rollback_dir"; then
        run_cmd mv "$rollback_dir" "$cloud_web_root"
    fi
    echo "ERROR: Nginx check/reload failed. Rolled back web root." >&2
    exit 1
fi

if [[ "$had_previous" == "1" ]]; then
    shopt -s nullglob
    rollback_dirs=("$web_parent/.${web_name}.rollback."*)
    shopt -u nullglob
    if (( ${#rollback_dirs[@]} > cloud_keep_rollbacks )); then
        if [[ "$cloud_use_sudo" == "1" ]]; then
            mapfile -t sorted_rollbacks < <(sudo ls -1dt "${rollback_dirs[@]}")
        else
            mapfile -t sorted_rollbacks < <(ls -1dt "${rollback_dirs[@]}")
        fi
        for ((i=cloud_keep_rollbacks; i<${#sorted_rollbacks[@]}; ++i)); do
            run_cmd rm -rf "${sorted_rollbacks[i]}"
        done
    fi
fi
CLOUD_SCRIPT_BODY

CLOUD_REMOTE_SCRIPT="/tmp/chromaprint3d-cloud-deploy.$$.sh"
scp -P "$CLOUD_SSH_PORT" "$CLOUD_SCRIPT" "$CLOUD_HOST:$CLOUD_REMOTE_SCRIPT"
rm -f "$CLOUD_SCRIPT"
ssh "${cloud_ssh_opts[@]}" "$CLOUD_HOST" \
    "bash '$CLOUD_REMOTE_SCRIPT' '$CLOUD_WEB_ROOT' '$CLOUD_UPLOAD_TAR' '$CLOUD_USE_SUDO' '$CLOUD_RELOAD_NGINX' '$CLOUD_KEEP_ROLLBACKS'; rm -f '$CLOUD_REMOTE_SCRIPT'"

info "Updating backend stack on home host..."
HOME_SCRIPT="$(mktemp /tmp/chromaprint3d-home-deploy.XXXXXX.sh)"
cat > "$HOME_SCRIPT" <<'HOME_SCRIPT_BODY'
#!/usr/bin/env bash
set -euo pipefail

home_deploy_dir="$1"
home_use_sudo="$2"
home_rollback_dir="$3"
home_health_timeout="$4"

if [[ -z "$home_deploy_dir" || "$home_deploy_dir" == "/" ]]; then
    echo "ERROR: Refusing to run docker compose in unsafe HOME_DEPLOY_DIR: '$home_deploy_dir'" >&2
    exit 1
fi
if ! [[ "$home_health_timeout" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: HOME_HEALTH_TIMEOUT_SECONDS must be a positive integer." >&2
    exit 1
fi

if docker compose version >/dev/null 2>&1; then
    compose_cmd=(docker compose)
elif command -v docker-compose >/dev/null 2>&1; then
    compose_cmd=(docker-compose)
else
    echo "ERROR: docker compose/docker-compose is not available." >&2
    exit 1
fi

run_compose() {
    if [[ "$home_use_sudo" == "1" ]]; then
        sudo "${compose_cmd[@]}" "$@"
    else
        "${compose_cmd[@]}" "$@"
    fi
}

run_docker() {
    if [[ "$home_use_sudo" == "1" ]]; then
        sudo docker "$@"
    else
        docker "$@"
    fi
}

wait_compose_ready() {
    local deadline=$((SECONDS + home_health_timeout))
    while (( SECONDS < deadline )); do
        mapfile -t container_ids < <(run_compose ps -q)
        if (( ${#container_ids[@]} == 0 )); then
            sleep 3
            continue
        fi

        local all_ready=1
        for cid in "${container_ids[@]}"; do
            local status
            local health
            status="$(run_docker inspect -f '{{.State.Status}}' "$cid" 2>/dev/null || true)"
            health="$(run_docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "$cid" 2>/dev/null || true)"
            if [[ "$status" != "running" ]]; then
                all_ready=0
                break
            fi
            if [[ "$health" != "none" && "$health" != "healthy" ]]; then
                all_ready=0
                break
            fi
        done

        if [[ "$all_ready" == "1" ]]; then
            return 0
        fi
        sleep 3
    done
    return 1
}

if [[ "$home_use_sudo" == "1" ]]; then
    sudo -v
fi

cd "$home_deploy_dir"
mkdir -p "$home_rollback_dir"
stamp="$(date +%Y%m%d%H%M%S)"
run_compose config > "$home_rollback_dir/compose.$stamp.yaml"
run_compose images > "$home_rollback_dir/images.$stamp.txt" || true
run_compose pull
run_compose up -d --remove-orphans
if ! wait_compose_ready; then
    echo "ERROR: Updated backend stack did not become ready in ${home_health_timeout}s." >&2
    run_compose ps || true
    run_compose logs --tail 200 || true
    exit 1
fi
run_compose ps
HOME_SCRIPT_BODY

HOME_REMOTE_SCRIPT="/tmp/chromaprint3d-home-deploy.$$.sh"
scp -P "$HOME_SSH_PORT" "$HOME_SCRIPT" "$HOME_HOST:$HOME_REMOTE_SCRIPT"
rm -f "$HOME_SCRIPT"
ssh "${home_ssh_opts[@]}" "$HOME_HOST" \
    "bash '$HOME_REMOTE_SCRIPT' '$HOME_DEPLOY_DIR' '$HOME_USE_SUDO' '$HOME_ROLLBACK_DIR' '$HOME_HEALTH_TIMEOUT_SECONDS'; rm -f '$HOME_REMOTE_SCRIPT'"

info "Split deployment completed successfully."
