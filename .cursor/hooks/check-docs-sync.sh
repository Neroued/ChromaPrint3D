#!/bin/bash
# stop hook: remind to sync docs when API/CLI/build files changed but docs weren't

input=$(cat)
status=$(echo "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status',''))" 2>/dev/null)

[ "$status" != "completed" ] && { echo '{}'; exit 0; }

project_dir="${CURSOR_PROJECT_DIR:-$(pwd)}"
cd "$project_dir" || { echo '{}'; exit 0; }

changed=$(git diff --name-only HEAD 2>/dev/null; git diff --cached --name-only 2>/dev/null)
[ -z "$changed" ] && { echo '{}'; exit 0; }

needs_docs=false

if echo "$changed" | grep -qE '(web/backend/.*\.(cpp|h|hpp))'; then
    needs_docs=true
fi
if echo "$changed" | grep -qE '(apps/.*\.(cpp|h|hpp))'; then
    needs_docs=true
fi
if echo "$changed" | grep -qE '(CMakeLists\.txt|Dockerfile|\.github/)'; then
    needs_docs=true
fi

if [ "$needs_docs" = false ]; then
    echo '{}'
    exit 0
fi

docs_updated=false
if echo "$changed" | grep -qE '(^docs/|^README\.md|^AGENTS\.md)'; then
    docs_updated=true
fi

if [ "$docs_updated" = false ]; then
    cat <<'EOF'
{"followup_message":"检测到 API/CLI/构建相关文件有变更，但 docs/ 和 README.md 未同步更新。请检查是否需要更新文档（参考 .cursor/rules/sync-docs.mdc 中的触发条件）。如果本次变更不涉及用户可见行为、API 参数或部署流程，可以忽略此提醒。"}
EOF
else
    echo '{}'
fi

exit 0
