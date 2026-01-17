#!/bin/bash
# =============================================================================
# Stage 2 Training: Vidarc Causal Fine-tuning with Self-Forcing
# =============================================================================
# Usage:
#   ./run_train_vidarc.sh [DATA_DIR] [OUTPUT_DIR] [MAX_STEPS]
#
# Example:
#   ./run_train_vidarc.sh ./data/vidarc_stack_bowls ./output_vidarc 4000
#
# Environment Variables:
#   VIDAR_ENV: Path to conda environment (default: self_forcing)
#   VIDAR_PATH: Path to vidar codebase (for wan modules)
# =============================================================================

# --- Environment Setup ---
# Use self_forcing environment (same as vidar server during inference)
# NOT RoboTwin-hb which is for simulation only
VIDAR_ENV=${VIDAR_ENV:-"/mnt/shared-storage-user/qinyiran/cyujie/cyujie/env/self_forcing"}

# Alternative environments (uncomment to use):
# VIDAR_ENV="/mnt/shared-storage-user/qinyiran/cyujie/cyujie/env/vidar"

echo "Activating conda environment: $VIDAR_ENV"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$VIDAR_ENV"

# Verify activation
if [ "$CONDA_PREFIX" != "$VIDAR_ENV" ]; then
    echo "ERROR: Failed to activate conda environment: $VIDAR_ENV"
    echo "Current CONDA_PREFIX: $CONDA_PREFIX"
    exit 1
fi

# Set library paths (optional, for mujoco if needed)
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:$LD_LIBRARY_PATH

# --- PYTHONPATH Setup ---
# Add vidar codebase to PYTHONPATH (for wan modules)
# NOTE: causal_worker.py does "import wan", so PYTHONPATH must include vidar dir
#       The self_forcing env should have easydict installed
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Vidar paths
VIDAR_LOCAL="$(dirname "$SCRIPT_DIR")/vidar"
VIDAR_SHARED="/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar"

# Priority order:
# 1. VIDAR_PATH environment variable (user override)
# 2. Shared storage vidar (same as inference causal_worker)
# 3. Local vidar
if [ -n "$VIDAR_PATH" ]; then
    export PYTHONPATH="$VIDAR_PATH:$PYTHONPATH"
    echo "Using VIDAR_PATH: $VIDAR_PATH"
elif [ -d "$VIDAR_SHARED/wan" ]; then
    export PYTHONPATH="$VIDAR_SHARED:$PYTHONPATH"
    echo "Using shared storage vidar: $VIDAR_SHARED"
elif [ -d "$VIDAR_LOCAL/wan" ]; then
    export PYTHONPATH="$VIDAR_LOCAL:$PYTHONPATH"
    echo "Using local vidar: $VIDAR_LOCAL"
else
    echo "ERROR: vidar codebase not found!"
    echo "  Set VIDAR_PATH environment variable or check these paths:"
    echo "  - $VIDAR_SHARED"
    echo "  - $VIDAR_LOCAL"
    exit 1
fi

# Add vidar-robotwin to PYTHONPATH
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "PYTHONPATH: $PYTHONPATH"

# --- Configuration ---
# Data paths
DATA_DIR=${1:-"./data/vidarc_stack_bowls"}
OUTPUT_DIR=${2:-"./output_vidarc"}
MAX_STEPS=${3:-4000}

# Model checkpoints
CKPT_DIR=${CKPT_DIR:-"/mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/vidar/Wan2.2-TI2V-5B"}
PT_DIR=${PT_DIR:-"/mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/vidar/vidar_ckpts/vidar.pt"}

# Training parameters
BATCH_SIZE=${BATCH_SIZE:-2}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-8}
CHUNK_SIZE=${CHUNK_SIZE:-16}
LR=${LR:-2e-5}
ETA=${ETA:-3.0}

# Distributed training
MASTER_PORT=${MASTER_PORT:-29500}
GPU_COUNT=$(nvidia-smi -L | wc -l)

# --- Validate Paths ---
echo "=========================================="
echo "Vidarc Stage 2 Training"
echo "=========================================="
echo "Data directory: $DATA_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Checkpoint directory: $CKPT_DIR"
echo "Stage 1 weights: $PT_DIR"
echo "GPUs: $GPU_COUNT"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION (effective: $((BATCH_SIZE * GRADIENT_ACCUMULATION * GPU_COUNT)))"
echo "Max steps: $MAX_STEPS"
echo "=========================================="

# Check if data exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo "Please prepare data first using:"
    echo "  python scripts/prepare_robotwin2.py --src-dir <source> --dst-dir $DATA_DIR"
    exit 1
fi

# Check if checkpoints exist
if [ ! -d "$CKPT_DIR" ]; then
    echo "WARNING: Checkpoint directory not found: $CKPT_DIR"
    echo "Please download Wan2.2-TI2V-5B model"
fi

if [ ! -f "$PT_DIR" ]; then
    echo "WARNING: Stage 1 weights not found: $PT_DIR"
    echo "Training will start from base Wan2.2 weights"
    PT_FLAG=""
else
    PT_FLAG="--pt-dir $PT_DIR"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# --- Launch Training ---
cd "$SCRIPT_DIR"

echo ""
echo "Starting training..."
echo ""

torchrun --nproc_per_node=$GPU_COUNT --master_port=$MASTER_PORT \
    scripts/train_vidarc.py \
    --config configs/vidarc_2xh200.yaml \
    --data-dir "$DATA_DIR" \
    --ckpt-dir "$CKPT_DIR" \
    $PT_FLAG \
    --output-dir "$OUTPUT_DIR" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation "$GRADIENT_ACCUMULATION" \
    --chunk-size "$CHUNK_SIZE" \
    --lr "$LR" \
    --eta "$ETA" \
    --log-interval 10 \
    --save-interval 500

echo "=========================================="
echo "Training finished."
echo "Output saved to: $OUTPUT_DIR"
echo "=========================================="
