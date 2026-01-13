#!/bin/bash
# 安装 RoboTwin-hb 环境的脚本，带重试机制

set -e

ENV_PREFIX="/mnt/shared-storage-user/qinyiran/cyujie/cyujie/env/RoboTwin-hb"
YAML_FILE="RoboTwin-hb-custom.yaml"
MAX_RETRIES=3

echo "=========================================="
echo "RoboTwin-hb 环境安装脚本"
echo "=========================================="
echo "环境路径: $ENV_PREFIX"
echo "配置文件: $YAML_FILE"
echo "最大重试次数: $MAX_RETRIES"
echo "=========================================="

# 检查 YAML 文件是否存在
if [ ! -f "$YAML_FILE" ]; then
    echo "错误: 找不到 $YAML_FILE"
    exit 1
fi

# 配置 conda 超时设置
echo "配置 conda 超时设置..."
conda config --set remote_connect_timeout_secs 60.0
conda config --set remote_read_timeout_secs 300.0

# 清理 conda 缓存（可选，但有助于解决网络问题）
echo "清理 conda 缓存..."
conda clean --all -y

# 尝试安装，带重试机制
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo ""
    echo "=========================================="
    echo "尝试安装 (第 $((RETRY_COUNT + 1)) 次 / 共 $MAX_RETRIES 次)"
    echo "=========================================="
    
    if conda env create --prefix "$ENV_PREFIX" --file "$YAML_FILE" 2>&1; then
        echo ""
        echo "=========================================="
        echo "✓ 环境安装成功！"
        echo "=========================================="
        echo "激活环境使用:"
        echo "  conda activate $ENV_PREFIX"
        echo "或者:"
        echo "  source $ENV_PREFIX/bin/activate"
        exit 0
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo ""
            echo "安装失败，等待 10 秒后重试..."
            sleep 10
            echo "清理部分下载的包..."
            conda clean --packages -y
        else
            echo ""
            echo "=========================================="
            echo "✗ 安装失败，已重试 $MAX_RETRIES 次"
            echo "=========================================="
            echo "建议："
            echo "1. 检查网络连接"
            echo "2. 尝试使用国内 conda 镜像源"
            echo "3. 手动分步安装依赖"
            exit 1
        fi
    fi
done

