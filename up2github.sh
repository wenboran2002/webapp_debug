#!/bin/bash

# 获取当前分支名称
BRANCH=$(git symbolic-ref --short HEAD)

echo "🚀 正在准备强制上传到远程分支: $BRANCH ..."

# 1. 强制添加所有文件 (包括删除的文件)
git add .

# 2. 提交 (如果没有传入参数，则使用默认时间戳作为 commit message)
MSG="$1"
if [ -z "$MSG" ]; then
    MSG="Auto-save: $(date '+%Y-%m-%d %H:%M:%S')"
fi

git commit -m "$MSG"

# 3. 强制推送 (覆盖远程仓库)
echo "🔥 正在强制推送 (Force Push)..."
git push --force origin "$BRANCH"

if [ $? -eq 0 ]; then
    echo "✅ 强制上传成功！远程仓库已被覆盖。"
else
    echo "❌ 上传失败，请检查网络或权限。"
fi