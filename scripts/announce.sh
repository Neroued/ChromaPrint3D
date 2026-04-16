#!/usr/bin/env bash
#
# ChromaPrint3D announcement management CLI.
#
# Usage:
#   ./scripts/announce.sh list
#   ./scripts/announce.sh create <id> <type> <severity> <ends_at> \
#       [--title-zh TEXT] [--title-en TEXT] \
#       [--body-zh TEXT]  [--body-en TEXT] \
#       [--starts-at ISO] [--scheduled-update-at ISO] \
#       [--dismissible true|false]
#   ./scripts/announce.sh delete <id>
#   ./scripts/announce.sh from-file <path.yaml|path.json>
#
# Environment (all optional, with sane defaults for local dev):
#   CHROMAPRINT3D_API_URL            default: https://chromaprint3d.com
#   CHROMAPRINT3D_ANNOUNCEMENT_TOKEN required for create/delete/from-file
#   CHROMAPRINT3D_ANNOUNCEMENT_TOKEN_FILE
#                                    alt. path to a file holding the token
#   CURL_EXTRA_ARGS                  extra args passed through to curl
#
# The script NEVER prints the token to stdout/stderr and only sends it
# via the `x-announcement-token` header, strictly as documented in the
# backend advice in web/backend/src/presentation/announcement_auth_advice.cpp.

set -euo pipefail

die()  { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*"; }

# ---------- Argument helpers ----------

sub_usage() {
    grep -E '^#' "$0" | sed -E 's/^# ?//'
    exit 0
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "missing required command: $1"
}

resolve_api_url() {
    local url="${CHROMAPRINT3D_API_URL:-https://chromaprint3d.com}"
    url="${url%/}"
    printf '%s' "$url"
}

resolve_token() {
    if [[ -n "${CHROMAPRINT3D_ANNOUNCEMENT_TOKEN:-}" ]]; then
        printf '%s' "$CHROMAPRINT3D_ANNOUNCEMENT_TOKEN"
        return 0
    fi
    if [[ -n "${CHROMAPRINT3D_ANNOUNCEMENT_TOKEN_FILE:-}" ]]; then
        [[ -f "$CHROMAPRINT3D_ANNOUNCEMENT_TOKEN_FILE" ]] ||
            die "token file not found: $CHROMAPRINT3D_ANNOUNCEMENT_TOKEN_FILE"
        tr -d '\r\n' <"$CHROMAPRINT3D_ANNOUNCEMENT_TOKEN_FILE"
        return 0
    fi
    die "announcement token not configured (set CHROMAPRINT3D_ANNOUNCEMENT_TOKEN)"
}

curl_with_header() {
    # Call curl with --fail-with-body so non-2xx responses still print body
    # without leaking the token via the command line.
    local token
    token="$(resolve_token)"
    local extra=()
    if [[ -n "${CURL_EXTRA_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        extra=($CURL_EXTRA_ARGS)
    fi
    curl --fail-with-body \
         --silent --show-error \
         -H "x-announcement-token: $token" \
         "${extra[@]}" \
         "$@"
}

# ---------- Commands ----------

cmd_list() {
    local api_url
    api_url="$(resolve_api_url)"
    # GET is public — no token needed.
    local extra=()
    if [[ -n "${CURL_EXTRA_ARGS:-}" ]]; then
        # shellcheck disable=SC2206
        extra=($CURL_EXTRA_ARGS)
    fi
    curl --fail-with-body --silent --show-error "${extra[@]}" \
         "$api_url/api/v1/announcements" |
        pretty_json
}

cmd_delete() {
    local id="${1:-}"
    [[ -n "$id" ]] || die "id is required"
    local api_url
    api_url="$(resolve_api_url)"
    curl_with_header -X DELETE "$api_url/api/v1/announcements/$id" |
        pretty_json
}

cmd_create() {
    local id="${1:-}" type="${2:-}" severity="${3:-}" ends_at="${4:-}"
    [[ -n "$id" && -n "$type" && -n "$severity" && -n "$ends_at" ]] ||
        die "usage: announce.sh create <id> <type> <severity> <ends_at> [flags]"
    shift 4

    local title_zh=""; local title_en=""
    local body_zh="";  local body_en=""
    local starts_at="" scheduled_update_at="" dismissible=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --title-zh) title_zh="$2"; shift 2;;
            --title-en) title_en="$2"; shift 2;;
            --body-zh)  body_zh="$2";  shift 2;;
            --body-en)  body_en="$2";  shift 2;;
            --starts-at) starts_at="$2"; shift 2;;
            --scheduled-update-at) scheduled_update_at="$2"; shift 2;;
            --dismissible) dismissible="$2"; shift 2;;
            *) die "unknown flag: $1";;
        esac
    done

    [[ -n "$title_zh" || -n "$title_en" ]] ||
        die "at least one of --title-zh or --title-en is required"
    [[ -n "$body_zh" || -n "$body_en" ]] ||
        die "at least one of --body-zh or --body-en is required"

    local payload
    payload="$(build_payload_json \
        "$id" "$type" "$severity" "$ends_at" \
        "$title_zh" "$title_en" \
        "$body_zh" "$body_en" \
        "$starts_at" "$scheduled_update_at" "$dismissible")"

    local api_url
    api_url="$(resolve_api_url)"
    printf '%s' "$payload" |
        curl_with_header -X POST \
            -H 'content-type: application/json' \
            --data-binary @- \
            "$api_url/api/v1/announcements" |
        pretty_json
}

cmd_from_file() {
    local path="${1:-}"
    [[ -n "$path" && -f "$path" ]] || die "from-file requires an existing file"

    local payload
    payload="$(render_payload_from_file "$path")"

    local api_url
    api_url="$(resolve_api_url)"
    printf '%s' "$payload" |
        curl_with_header -X POST \
            -H 'content-type: application/json' \
            --data-binary @- \
            "$api_url/api/v1/announcements" |
        pretty_json
}

# ---------- JSON builders ----------

pretty_json() {
    if command -v jq >/dev/null 2>&1; then
        jq .
    elif command -v python3 >/dev/null 2>&1; then
        python3 -m json.tool --no-ensure-ascii 2>/dev/null || python3 -m json.tool
    else
        cat
    fi
}

build_payload_json() {
    # Delegates to Python for safe JSON escaping without depending on jq.
    python3 - "$@" <<'PY'
import json
import sys

(_, id_, type_, severity, ends_at,
 title_zh, title_en, body_zh, body_en,
 starts_at, scheduled_update_at, dismissible) = sys.argv

payload = {
    "id": id_,
    "type": type_,
    "severity": severity,
    "title": {
        "zh": title_zh or None,
        "en": title_en or None,
    },
    "body": {
        "zh": body_zh or None,
        "en": body_en or None,
    },
    "ends_at": ends_at,
}
if starts_at:
    payload["starts_at"] = starts_at
if scheduled_update_at:
    payload["scheduled_update_at"] = scheduled_update_at
if dismissible:
    if dismissible.lower() in ("true", "1", "yes"):
        payload["dismissible"] = True
    elif dismissible.lower() in ("false", "0", "no"):
        payload["dismissible"] = False
    else:
        sys.exit(f"invalid --dismissible: {dismissible}")

print(json.dumps(payload, ensure_ascii=False))
PY
}

render_payload_from_file() {
    local path="$1"
    python3 - "$path" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "rb") as fh:
    raw = fh.read()

data = None
ext = path.lower().rsplit(".", 1)[-1] if "." in path else ""
if ext in ("yaml", "yml"):
    try:
        import yaml  # type: ignore
    except ImportError:
        sys.exit(
            "ERROR: PyYAML is required for YAML input — "
            "install it with `pip install pyyaml` or convert to JSON."
        )
    data = yaml.safe_load(raw)
else:
    data = json.loads(raw)

if not isinstance(data, dict):
    sys.exit("ERROR: announcement payload must be a JSON/YAML object")

# Normalize title/body helpers: accept flat title_zh / title_en too.
def merge_lang(obj, key):
    obj.setdefault(key, {})
    zh = obj.pop(f"{key}_zh", None)
    en = obj.pop(f"{key}_en", None)
    if zh is not None:
        obj[key]["zh"] = zh
    if en is not None:
        obj[key]["en"] = en

merge_lang(data, "title")
merge_lang(data, "body")

print(json.dumps(data, ensure_ascii=False))
PY
}

# ---------- Entry ----------

main() {
    require_cmd curl
    require_cmd python3

    local cmd="${1:-}"
    case "$cmd" in
        list)           shift; cmd_list "$@";;
        create)         shift; cmd_create "$@";;
        delete)         shift; cmd_delete "$@";;
        from-file)      shift; cmd_from_file "$@";;
        -h|--help|help) sub_usage;;
        "")             sub_usage;;
        *)              die "unknown command: $cmd (try --help)";;
    esac
}

main "$@"
