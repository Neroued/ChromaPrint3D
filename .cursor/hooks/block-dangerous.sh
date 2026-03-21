#!/bin/bash
# beforeShellExecution hook: block destructive commands

input=$(cat)
command=$(echo "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('command',''))" 2>/dev/null)

[ -z "$command" ] && { echo '{"permission":"allow"}'; exit 0; }

deny_with() {
    cat <<EOF
{"permission":"deny","user_message":"$1","agent_message":"$1 请使用更安全的替代方案。"}
EOF
    exit 0
}

ask_with() {
    cat <<EOF
{"permission":"ask","user_message":"$1"}
EOF
    exit 0
}

# --- hard block ---

if echo "$command" | grep -qE 'rm\s+(-[^ ]*)?r[^ ]*f[^ ]*\s+/\s*$|rm\s+(-[^ ]*)?r[^ ]*f[^ ]*\s+/\*'; then
    deny_with "已拦截：递归删除根文件系统"
fi

if echo "$command" | grep -qE 'git\s+push\s+.*--force.*\s+(origin\s+)?(master|main)\b|git\s+push\s+-f\s+.*\s+(origin\s+)?(master|main)\b'; then
    deny_with "已拦截：禁止 force push 到 master/main"
fi

if echo "$command" | grep -qE 'mkfs\.|>\s*/dev/sd|:\(\)\{|fork\s*bomb'; then
    deny_with "已拦截：潜在的破坏性系统命令"
fi

if echo "$command" | grep -qE 'dd\s+.*of=/dev/sd'; then
    deny_with "已拦截：直接写入块设备"
fi

# --- ask confirmation ---

if echo "$command" | grep -qE 'git\s+reset\s+--hard'; then
    ask_with "git reset --hard 将丢弃所有未提交的更改，是否继续？"
fi

if echo "$command" | grep -qE 'git\s+clean\s+-[^ ]*f'; then
    ask_with "git clean -f 将删除未跟踪的文件，是否继续？"
fi

echo '{"permission":"allow"}'
exit 0
