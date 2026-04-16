#!/bin/bash
# beforeReadFile hook: block reading sensitive files

input=$(cat)
file_path=$(echo "$input" | python3 -c "import sys,json; print(json.load(sys.stdin).get('file_path',''))" 2>/dev/null)

[ -z "$file_path" ] && { echo '{"permission":"allow"}'; exit 0; }

filename=$(basename "$file_path")

deny_with() {
    cat <<EOF
{"permission":"deny","user_message":"已阻止读取敏感文件: $filename — $1"}
EOF
    exit 0
}

case "$filename" in
    .env|.env.local|.env.production|.env.staging)
        deny_with "环境变量文件可能包含密钥"
        ;;
    .env.*.local)
        deny_with "本地环境变量文件可能包含密钥"
        ;;
    announcement.env|.chromaprint3d-announcement-token)
        deny_with "公告 token 文件"
        ;;
    credentials.json|service-account*.json)
        deny_with "凭据文件"
        ;;
    id_rsa|id_ed25519|id_ecdsa|*.pem)
        deny_with "私钥文件"
        ;;
    .git-credentials|.netrc)
        deny_with "Git 凭据文件"
        ;;
esac

case "$file_path" in
    */.ssh/*)
        deny_with "SSH 目录中的文件"
        ;;
esac

echo '{"permission":"allow"}'
exit 0
