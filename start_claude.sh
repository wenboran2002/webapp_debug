#!/bin/bash

# 1. 基础 API 配置
export ANTHROPIC_AUTH_TOKEN="sk-fYvlXvBzdMK0TLzHsG6kGJB9mjxLl8GecqrxoqcjaWV1h1pN"
export ANTHROPIC_BASE_URL="https://anyrouter.top"

# 2. 定义清理函数：取消代理设置
cleanup() {
    unset http_proxy
    unset https_proxy
    # echo "Proxy unset completed." # 如果你想看到提示可以取消注释这行
}

# 3. 设置 trap：当脚本退出(EXIT)或被中断(INT, 如Ctrl+C)时，自动执行 cleanup
trap cleanup EXIT INT

# 4. 设置代理
export http_proxy=http://127.0.0.1:7865
export https_proxy=http://127.0.0.1:7865

# 5. 启动 Claude
claude "$@"