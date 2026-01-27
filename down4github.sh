#!/bin/bash

# 获取当前分支名称
BRANCH=$(git symbolic-ref --short HEAD)

echo "⚠️  正在准备从远程分支: $BRANCH 强制拉取..."
echo "⚠️  警告：本地所有未提交的修改都将永久丢失！"

# 1. 获取远程最新状态
git fetch --all

# 2. 强制重置本地代码与远程保持一致
echo "🔥 正在执行 Hard Reset..."
git reset --hard "origin/$BRANCH"

if [ $? -eq 0 ]; then
    echo "✅ 强制下载成功！本地代码已与远程完全一致。"
else
    echo "❌ 下载失败，请检查网络。"
fi