#!/bin/bash
# afterFileEdit hook: auto-format C++ files with clang-format

input=$(cat)
file_path=$(echo "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('file_path',''))" 2>/dev/null)

[ -z "$file_path" ] && exit 0
[ ! -f "$file_path" ] && exit 0

case "$file_path" in
    *.cpp|*.h|*.hpp|*.cc|*.cxx)
        if command -v clang-format &>/dev/null; then
            clang-format -i "$file_path" 2>/dev/null
        fi
        ;;
esac

exit 0
