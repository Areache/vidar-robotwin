#!/bin/bash
# Train Vidarc (Stage 2) training script wrapper
# This script ensures the correct Python environment is used, consistent with run_eval_ddp_causal.sh

# Set conda environment path
CONDA_ENV="/mnt/shared-storage-user/qinyiran/cyujie/cyujie/env/RoboTwin-hb"

# Initialize conda (if not already initialized)
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    # Try to source conda.sh if available
    if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
        source "$HOME/miniconda3/etc/profile.d/conda.sh"
    elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
        source "/opt/conda/etc/profile.d/conda.sh"
    fi
fi

# Activate conda environment (same as run_eval_ddp_causal.sh)
conda activate "$CONDA_ENV"

# Set working directory
cd /mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar-robotwin

# Export environment variables if needed
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH

# Ensure we use the conda environment's Python and torchrun
export PATH="$CONDA_ENV/bin:$PATH"

# Default parameters
# Parameter order: [CONFIG] [DATA_DIR] [CKPT_DIR] [PT_DIR] [OUTPUT_DIR] [MAX_STEPS] [NPROC] [PORT]
# If first arg doesn't end with .yaml, assume it's DATA_DIR (skip CONFIG, use default)

# Check if first argument is a config file
if [ -n "$1" ] && [[ "$1" == *.yaml ]] && [ -f "$1" ]; then
    # First arg is config file
    CONFIG="$1"
    DATA_DIR=${2:-"/mnt/shared-storage-user/qinyiran/cyujie/cyujie/datasets/RoboTwin2.0/dataset/stack_bowls_two"}
    CKPT_DIR=${3:-"/mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/vidar/Wan2.2-TI2V-5B"}
    PT_DIR=${4:-"/mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/vidar/vidar_ckpts/vidar.pt"}
    OUTPUT_DIR=${5:-"./output_vidarc"}
    MAX_STEPS=${6:-4000}
else
    # First arg is DATA_DIR (no config provided, use default)
    CONFIG="configs/vidarc_2xh200.yaml"
    DATA_DIR=${1:-"/mnt/shared-storage-user/qinyiran/cyujie/cyujie/datasets/RoboTwin2.0/dataset/stack_bowls_two"}
    CKPT_DIR=${2:-"/mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/vidar/Wan2.2-TI2V-5B"}
    PT_DIR=${3:-"/mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/vidar/vidar_ckpts/vidar.pt"}
    OUTPUT_DIR=${4:-"./output_vidarc"}
    MAX_STEPS=${5:-4000}
fi

NPROC_PER_NODE=${6:-$(nvidia-smi -L | wc -l)}
MASTER_PORT=${7:-29500}

echo "=========================================="
echo "Starting Vidarc Stage 2 Training..."
echo "=========================================="
echo "Config: $CONFIG"
echo "Data dir: $DATA_DIR"
echo "Checkpoint dir: $CKPT_DIR"
echo "Stage 1 weights: $PT_DIR"
echo "Output dir: $OUTPUT_DIR"
echo "Max steps: $MAX_STEPS"
echo "GPUs: $NPROC_PER_NODE"
echo "=========================================="

# Use the conda environment's Python to run torchrun
# This ensures torchrun uses the correct Python interpreter
PYTHON_BIN="$CONDA_ENV/bin/python"
TORCHRUN_BIN="$CONDA_ENV/bin/torchrun"

# Check if torchrun exists in conda environment, otherwise use Python module
if [ -f "$TORCHRUN_BIN" ]; then
    TORCHRUN_CMD="$TORCHRUN_BIN"
else
    TORCHRUN_CMD="$PYTHON_BIN -m torch.distributed.run"
fi

export MASTER_PORT=$MASTER_PORT
echo "Using Python: $PYTHON_BIN"
echo "Using torchrun: $TORCHRUN_CMD"
echo "=========================================="

$TORCHRUN_CMD --nproc_per_node=$NPROC_PER_NODE --master_port=$MASTER_PORT \
    scripts/train_vidarc.py \
    --config "$CONFIG" \
    --data-dir "$DATA_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    --pt-dir "$PT_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS"

echo "=========================================="
echo "Training finished."
echo "=========================================="

