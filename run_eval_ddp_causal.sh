#!/bin/bash
conda activate /mnt/shared-storage-user/qinyiran/cyujie/cyujie/env/RoboTwin-hb
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH
# --- 配置区域 ---
# 任务配置
TASK_CONFIG=${1:-"hd_clean"}

# 模型路径配置
# 优先使用本地路径，如果不存在则从对象存储下载
LOCAL_MODEL_DIR="/mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/vidar/vidar_ckpts"
OSS_MODEL_PATH="h-ceph:qinyiran/vidar/vidar_ckpts"

# 确保本地目录存在
mkdir -p "$LOCAL_MODEL_DIR"

# 检查文件是否存在
if [ -f "$LOCAL_MODEL_DIR/vidarc.pt" ]; then
    DEFAULT_MODEL="$LOCAL_MODEL_DIR/vidarc.pt"
    DEFAULT_IDM="$LOCAL_MODEL_DIR/idm.pt"
    echo "使用本地模型文件: $DEFAULT_MODEL"
else
    echo "错误: 模型文件不存在，请检查路径或手动下载"
    exit 1
fi

MODEL=${2:-"$DEFAULT_MODEL"}
IDM=${3:-"$DEFAULT_IDM"} 
PREFIX=${4:-"ddp_causal"}

# 采样参数
NUM_NEW_FRAMES=${5:-16}
NUM_SAMPLING_STEP=${6:-10}
CFG=${7:-3.0}

# Video model subgoal 参数
USE_VIDEO_SUBGOALS=${8:-true}
# 对于 v2_mpc 和 df 版本，禁用 subgoal
if [ "$VERSION" = "v2_mpc" ] || [ "$VERSION" = "df" ]; then
    USE_VIDEO_SUBGOALS=false
    USE_LIBERO_SUBGOAL=false
else
    USE_VIDEO_SUBGOALS=true
    USE_LIBERO_SUBGOAL=true
fi
VIDEO_MODEL_CONFIG_PATH=${9:-"../vidar/vm/config/libero/lb_tk8_65to72.py"}  # 配置文件路径，例如: "config/libero/lb_tk8_65to72.py"
USE_VID_FIRST_N_FRAMES=${10:-2}
NUM_VID_PRED_PER_EP=${11:-5}

# Version parameter (v0_original, v1_subgoal, or v2_mpc)
VERSION=${12:-"v0_original"}  # Default to v0_original

# MPC parameters (for v2_mpc version)
MPC_NUM_CANDIDATES=${13:-50}  # Number of MPC candidates
MPC_COST_WEIGHTS=${14:-'{"task":1.0,"ctrl":0.1,"reach":0.5}'}  # MPC cost weights in JSON format

# Server 脚本位置 (根据需要修改，支持 T2V 或 I2V)
SERVER_SCRIPT="../vidar/server/causal_worker.sh"
OUTPUT_DIR="eval_result_test/ar"

# LIBERO HTTP 服务器配置
LIBERO_SERVER_PORT=${LIBERO_SERVER_PORT:-25401}
LIBERO_SERVER_SCRIPT="../vidar/vm/diffuser/libero/start_libero_subgoal_server.sh"
LIBERO_SERVER_URL="http://localhost:${LIBERO_SERVER_PORT}"

# --- 启动 LIBERO HTTP 服务器（如果需要） ---
# 检查是否需要启动 LIBERO 服务器（根据配置判断）
# 这里假设如果使用 HTTP 模式，需要启动服务器
check_libero_server() {
    if curl -s "$LIBERO_SERVER_URL/" > /dev/null 2>&1; then
        echo "LIBERO HTTP server is already running on port $LIBERO_SERVER_PORT"
        return 0
    else
        return 1
    fi
}

start_libero_server() {
    echo "=========================================="
    echo "Starting LIBERO Subgoal HTTP Server..."
    echo "=========================================="
    
    # 设置服务器环境变量
    export LIBERO_LOGBASE=${LIBERO_LOGBASE:-"/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm/logs"}
    export LIBERO_DATASET=${LIBERO_DATASET:-"libero-8tk-65to72-v3"}
    export LIBERO_EXP_NAME=${LIBERO_EXP_NAME:-"lb_tk8_65to72"}
    export LIBERO_EPOCH=${LIBERO_EPOCH:-"latest"}
    export LIBERO_SERVER_PORT=$LIBERO_SERVER_PORT
    export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-"/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm/config/libero/lb_tk8_65to72.py"}
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
    export PYTHONPATH=/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm:$PYTHONPATH
    
    # 在后台启动服务器（脚本会在 vm 根目录运行）
    cd /mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm/diffuser/libero
    nohup bash start_libero_subgoal_server.sh > /tmp/libero_server.log 2>&1 &
    LIBERO_SERVER_PID=$!
    echo "LIBERO server started with PID: $LIBERO_SERVER_PID"
    echo "Server logs: /tmp/libero_server.log"
    echo "You can check server status with: curl http://localhost:${LIBERO_SERVER_PORT}/"
    
    # 等待服务器启动（最多等待 120 秒，因为模型加载需要时间）
    echo "Waiting for server to be ready (model loading may take 1-2 minutes)..."
    for i in {1..120}; do
        if curl -s "$LIBERO_SERVER_URL/" > /dev/null 2>&1; then
            # 检查模型是否已初始化
            response=$(curl -s "$LIBERO_SERVER_URL/")
            if echo "$response" | grep -q '"model_initialized":true'; then
                echo "✓ LIBERO server is ready and model is initialized!"
                return 0
            elif echo "$response" | grep -q '"model_initialized":false'; then
                if [ $i -lt 120 ]; then
                    # 模型还在加载中
                    if [ $((i % 10)) -eq 0 ]; then
                        echo "  Model is still loading... ($i/120 seconds)"
                        echo "  Check logs: tail -f /tmp/libero_server.log"
                    fi
                else
                    echo "⚠ Warning: Server is running but model initialization failed"
                    echo "  Check logs: tail -20 /tmp/libero_server.log"
                    return 1
                fi
            fi
        fi
        sleep 1
        if [ $((i % 10)) -eq 0 ]; then
            echo "  Still waiting... ($i/120 seconds)"
        fi
    done
    
    echo "⚠ Warning: LIBERO server may not be ready yet, continuing anyway..."
    echo "  Check logs: tail -20 /tmp/libero_server.log"
    return 1
}

# 检查并启动 LIBERO 服务器（v2_mpc 和 df 版本不需要）
if [ "$VERSION" = "v2_mpc" ] || [ "$VERSION" = "df" ]; then
    echo "=========================================="
    if [ "$VERSION" = "v2_mpc" ]; then
        echo "v2_mpc version: Skipping LIBERO server (not needed for MPC)"
    else
        echo "df version: Skipping LIBERO server (not needed for diffusion forcing)"
    fi
    echo "=========================================="
    LIBERO_SERVER_STARTED=1  # 标记为未启动
elif ! check_libero_server; then
    start_libero_server
    LIBERO_SERVER_STARTED=$?
else
    LIBERO_SERVER_STARTED=0
fi

# --- 启动 ---
echo "=========================================="
echo "Starting Unified DDP Evaluation..."
echo "=========================================="
echo "Model: $MODEL"
echo "Prefix: $PREFIX"
echo "Server: $SERVER_SCRIPT"
echo "Version: $VERSION"
if [ "$VERSION" != "v2_mpc" ] && [ "$VERSION" != "df" ] && [ $LIBERO_SERVER_STARTED -eq 0 ]; then
    echo "LIBERO HTTP Server: $LIBERO_SERVER_URL (running)"
fi

export MASTER_PORT=11452
GPU_COUNT=$(nvidia-smi -L | wc -l)
cd /mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar-robotwin
# Export MPC parameters as environment variables (for v2_mpc version)
export MPC_NUM_CANDIDATES="$MPC_NUM_CANDIDATES"
export MPC_COST_WEIGHTS="$MPC_COST_WEIGHTS"

torchrun --nproc_per_node=$GPU_COUNT --master_port=$MASTER_PORT \
    policy/AR/run_eval_ddp.py \
    --server_script "$SERVER_SCRIPT" \
    --output_dir "$OUTPUT_DIR" \
    --model "$MODEL" \
    --idm "$IDM" \
    --prefix "$PREFIX" \
    --task_config "$TASK_CONFIG" \
    --num_new_frames "$NUM_NEW_FRAMES" \
    --num_sampling_step "$NUM_SAMPLING_STEP" \
    --cfg "$CFG" \
    --version "$VERSION" \
    --use_video_subgoals "$USE_VIDEO_SUBGOALS" \
    --use_libero_subgoal "$USE_LIBERO_SUBGOAL" \
    ${VIDEO_MODEL_CONFIG_PATH:+--video_model_config_path "$VIDEO_MODEL_CONFIG_PATH"} \
    --use_vid_first_n_frames "$USE_VID_FIRST_N_FRAMES" \
    --num_vid_pred_per_ep "$NUM_VID_PRED_PER_EP" \
    ${MPC_NUM_CANDIDATES:+--mpc_num_candidates "$MPC_NUM_CANDIDATES"} \
    ${MPC_COST_WEIGHTS:+--mpc_cost_weights "$MPC_COST_WEIGHTS"} \
    ${TASK_NAME:+--task_name "$TASK_NAME"}

echo "=========================================="
echo "Evaluation finished."
echo "=========================================="

# 清理：如果我们在脚本中启动了 LIBERO 服务器，可以选择关闭它
# 注释掉下面的代码以保持服务器运行（用于多次评估）
# if [ ! -z "$LIBERO_SERVER_PID" ]; then
#     echo "Stopping LIBERO server (PID: $LIBERO_SERVER_PID)..."
#     kill $LIBERO_SERVER_PID 2>/dev/null || true
#     echo "LIBERO server stopped."
# fi

