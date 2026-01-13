# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 切换到项目根目录
cd "$PROJECT_ROOT"

mkdir -p assets

# 设置 rclone 配置（尝试多个可能的路径）
RCLONE_CONFIG="${RCLONE_CONFIG:-}"

# 如果未设置，尝试自动查找
if [ -z "$RCLONE_CONFIG" ]; then
    # 方法1: 使用 rclone config file 命令获取
    CONFIG_FILE=$(rclone config file 2>/dev/null | grep -i "configuration file" | awk -F: '{print $2}' | xargs)
    if [ -n "$CONFIG_FILE" ] && [ -f "$CONFIG_FILE" ]; then
        RCLONE_CONFIG="$CONFIG_FILE"
    else
        # 方法2: 尝试常见的配置文件路径
        for config_path in \
            "/root/.config/rclone/rclone.conf" \
            "$HOME/.config/rclone/rclone.conf" \
            "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/.config/rclone/rclone.conf"; do
            if [ -f "$config_path" ]; then
                RCLONE_CONFIG="$config_path"
                break
            fi
        done
    fi
fi

export RCLONE_CONFIG

# 检查配置文件是否存在
if [ -z "$RCLONE_CONFIG" ] || [ ! -f "$RCLONE_CONFIG" ]; then
    echo "错误: rclone 配置文件不存在"
    echo ""
    echo "请执行以下操作之一:"
    echo "  1. 设置环境变量: export RCLONE_CONFIG=/path/to/rclone.conf"
    echo "  2. 配置 rclone: rclone config"
    echo ""
    echo "尝试的路径:"
    echo "  /root/.config/rclone/rclone.conf"
    echo "  $HOME/.config/rclone/rclone.conf"
    echo ""
    echo "当前用户: $(whoami)"
    echo "HOME 目录: $HOME"
    exit 1
fi

# 检查 h-ceph 配置是否存在
if ! rclone config show h-ceph &> /dev/null; then
    echo "错误: h-ceph 配置不存在"
    echo "配置文件: $RCLONE_CONFIG"
    echo "请先配置: rclone config"
    echo "或检查配置文件中是否有 [h-ceph] 部分"
    exit 1
fi

echo "使用配置文件: $RCLONE_CONFIG"
echo "项目根目录: $PROJECT_ROOT"

# 临时目录用于下载 zip 文件
# 使用项目根目录下的临时目录，避免 /tmp 空间不足
TEMP_DIR="$PROJECT_ROOT/.robotwin_assets_temp_$$"
mkdir -p "$TEMP_DIR"
echo "临时目录: $TEMP_DIR"

# 检查临时目录空间
TEMP_SPACE=$(df -BG "$TEMP_DIR" 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')
if [ -n "$TEMP_SPACE" ] && [ "$TEMP_SPACE" -lt 20 ]; then
    echo "警告: 临时目录可用空间不足 20GB (当前: ${TEMP_SPACE}GB)"
    echo "需要至少 15GB 空间来下载和解压文件"
    read -p "继续? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        exit 1
    fi
fi

echo "从对象存储下载资源文件..."

# 从对象存储下载 zip 文件到临时目录
echo "下载 background_texture.zip..."
if ! rclone copy \
    h-ceph:qinyiran/vidar-robotwin/TianxingChen/RoboTwin2.0/background_texture.zip \
    "$TEMP_DIR/" \
    --progress; then
    echo "错误: background_texture.zip 下载失败"
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "下载 embodiments.zip..."
if ! rclone copy \
    h-ceph:qinyiran/vidar-robotwin/TianxingChen/RoboTwin2.0/embodiments.zip \
    "$TEMP_DIR/" \
    --progress; then
    echo "错误: embodiments.zip 下载失败"
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "下载 objects.zip..."
if ! rclone copy \
    h-ceph:qinyiran/vidar-robotwin/TianxingChen/RoboTwin2.0/objects.zip \
    "$TEMP_DIR/" \
    --progress; then
    echo "错误: objects.zip 下载失败"
    rm -rf "$TEMP_DIR"
    exit 1
fi

echo "解压资源文件..."
echo "临时目录内容:"
ls -lh "$TEMP_DIR/" || echo "临时目录为空或不存在"

# 切换到 assets 目录进行解压
cd assets

# 解压文件
if [ -f "$TEMP_DIR/background_texture.zip" ]; then
    echo "解压 background_texture.zip..."
    unzip -q "$TEMP_DIR/background_texture.zip" || echo "警告: background_texture.zip 解压失败"
else
    echo "错误: background_texture.zip 不存在于 $TEMP_DIR/"
    echo "临时目录内容:"
    ls -lh "$TEMP_DIR/" 2>/dev/null || echo "临时目录不存在"
    rm -rf "$TEMP_DIR"
    exit 1
fi

if [ -f "$TEMP_DIR/embodiments.zip" ]; then
    echo "解压 embodiments.zip..."
    unzip -q "$TEMP_DIR/embodiments.zip" || echo "警告: embodiments.zip 解压失败"
else
    echo "错误: embodiments.zip 不存在于 $TEMP_DIR/"
    rm -rf "$TEMP_DIR"
    exit 1
fi

if [ -f "$TEMP_DIR/objects.zip" ]; then
    echo "解压 objects.zip..."
    unzip -q "$TEMP_DIR/objects.zip" || echo "警告: objects.zip 解压失败"
else
    echo "错误: objects.zip 不存在于 $TEMP_DIR/"
    rm -rf "$TEMP_DIR"
    exit 1
fi

# 清理临时文件
rm -rf "$TEMP_DIR"
echo "临时文件已清理"

# 返回项目根目录
cd "$PROJECT_ROOT"
echo "making vidar related file"
cp -r assets/embodiments/aloha-agilex assets/embodiments/aloha-vidar
cp vidar_assets/*.yml  assets/embodiments/aloha-vidar
cp vidar_assets/arx5_description_isaac.urdf assets/embodiments/aloha-vidar/urdf


echo "Configuring Path ..."
python ./script/update_embodiment_config_path.py