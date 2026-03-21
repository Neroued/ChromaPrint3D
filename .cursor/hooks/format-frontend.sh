#!/bin/bash
# afterFileEdit hook: auto-format Vue/TS/TSX files with prettier

input=$(cat)
file_path=$(echo "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('file_path',''))" 2>/dev/null)

[ -z "$file_path" ] && exit 0
[ ! -f "$file_path" ] && exit 0

case "$file_path" in
    */web/frontend/*.vue|*/web/frontend/*.ts|*/web/frontend/*.tsx)
        frontend_dir="${CURSOR_PROJECT_DIR:-$(pwd)}/web/frontend"
        if [ -d "$frontend_dir" ] && [ -x "$frontend_dir/node_modules/.bin/prettier" ]; then
            cd "$frontend_dir" && npx prettier --write "$file_path" 2>/dev/null
        fi
        ;;
esac

exit 0
