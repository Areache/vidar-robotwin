#!/bin/bash
# =============================================================================
# Stage 2 Training: Vidarc Causal Fine-tuning with Self-Forcing
# =============================================================================
# Usage:
#   ./run_train_vidarc.sh CONFIG DATA_DIR CKPT_DIR PT_DIR OUTPUT_DIR MAX_STEPS [RESUME]
#
# Example (training only):
#   ./run_train_vidarc.sh \
#       configs/vidarc_2xh200.yaml \
#       ./data/vidarc_stack_bowls \
#       /path/to/Wan2.2-TI2V-5B \
#       /path/to/vidar.pt \
#       ./output_vidarc \
#       4000
#
# Example (resume training):
#   ./run_train_vidarc.sh \
#       configs/vidarc_2xh200.yaml \
#       ./data/vidarc_stack_bowls \
#       /path/to/Wan2.2-TI2V-5B \
#       /path/to/vidar.pt \
#       ./output_vidarc \
#       4000 \
#       ./output_vidarc/checkpoints/vidar/vidarc_4x.pt
#
#   # Or using environment variable:
#   RESUME=./output_vidarc/checkpoints/vidar/vidarc_4x.pt \
#   ./run_train_vidarc.sh configs/vidarc_2xh200.yaml ... 4000
#
# Example (training + evaluation):
#   RUN_EVAL=true \
#   EVAL_TASK_NAME=adjust_bottle \
#   EVAL_CKPT=./checkpoints/vidar/vidarc_4x.pt \
#   ./run_train_vidarc.sh configs/vidarc_2xh200.yaml ...
#
# Environment Variables:
#   VIDAR_ENV: Path to conda environment (default: self_forcing)
#   VIDAR_PATH: Path to vidar codebase (for wan modules)
#   RESUME: Path to checkpoint file to resume from (optional, can also be 7th positional arg or in config file)
#
# Post-Training Evaluation (runs once at the end):
#   RUN_EVAL: Set to "true" to run evaluation after training completes (default: false)
#   EVAL_CKPT: Path to trained checkpoint (default: ./checkpoints/vidar/vidarc_4x.pt)
#
# In-Training Evaluation (runs after each checkpoint save):
#   RUN_EVAL_AFTER_SAVE: Set to "true" to evaluate after each checkpoint (default: false)
#   EVAL_TASK_NAME: Task name for evaluation (default: adjust_bottle)
#   EVAL_TASK_CONFIG: Task config (default: hd_clean)
#   EVAL_IDM: Path to IDM model (default: vidar_ckpts/idm.pt)
#   EVAL_PREFIX: Output prefix (default: step_<current_step>)
#   EVAL_NUM_NEW_FRAMES: Frames to generate (default: 16)
#   EVAL_NUM_SAMPLING_STEP: Sampling steps (default: 10)
#   EVAL_CFG: CFG scale (default: 3.0)
#
# Example (evaluate after each checkpoint):
#   RUN_EVAL_AFTER_SAVE=true \
#   EVAL_TASK_NAME=adjust_bottle \
#   ./run_train_vidarc.sh configs/vidarc_2xh200.yaml ...
#
# LoRA Training (memory-efficient finetuning):
#   USE_LORA: Set to "true" to enable LoRA (default: false)
#   LORA_RANK: LoRA rank, higher = more capacity (default: 32)
#   LORA_ALPHA: LoRA alpha scaling factor (default: 32)
#   LORA_TARGET_MODULES: Comma-separated modules (default: q,k,v,o)
#
# Example (LoRA training - reduces VRAM by ~75%):
#   USE_LORA=true LORA_RANK=32 \
#   ./run_train_vidarc.sh configs/vidarc_2xh200.yaml ...
#
# Few-Step Diffusion & Stochastic Gradient Truncation:
#   STOCHASTIC_TRUNCATION: Set to "true" to enable, "false" to disable
#   TRUNCATION_STRATEGY: "uniform", "importance", or "stratified" (default: uniform)
#   NUM_INFERENCE_STEPS: Number of diffusion steps (default: 10)
#
# Example (enable stochastic truncation with importance sampling):
#   STOCHASTIC_TRUNCATION=true TRUNCATION_STRATEGY=importance \
#   ./run_train_vidarc.sh configs/vidarc_2xh200.yaml ...
#
# Example (disable stochastic truncation for standard training):
#   STOCHASTIC_TRUNCATION=false \
#   ./run_train_vidarc.sh configs/vidarc_2xh200.yaml ...
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

# --- NCCL and Performance Environment Variables ---
# Increase NCCL timeout during debugging
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-1800}

# Better NCCL performance for 8 GPUs
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-NVL}

# Increase dynamo cache for flex_attention shapes
export TORCH_DYNAMO_CACHE_SIZE_LIMIT=${TORCH_DYNAMO_CACHE_SIZE_LIMIT:-128}

echo "NCCL_TIMEOUT: $NCCL_TIMEOUT"
echo "NCCL_IB_DISABLE: $NCCL_IB_DISABLE"
echo "NCCL_P2P_LEVEL: $NCCL_P2P_LEVEL"
echo "TORCH_DYNAMO_CACHE_SIZE_LIMIT: $TORCH_DYNAMO_CACHE_SIZE_LIMIT"

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
# Positional arguments (matching README_train.md documentation)
CONFIG=${1:-"configs/vidarc_2xh200.yaml"}
DATA_DIR=${2:-"./data/vidarc_stack_bowls"}
CKPT_DIR=${3:-"/mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/vidar/Wan2.2-TI2V-5B"}
PT_DIR=${4:-"/mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/vidar/vidar_ckpts/vidar.pt"}
OUTPUT_DIR=${5:-"./output_vidarc"}
MAX_STEPS=${6:-4000}
# Resume checkpoint: environment variable takes precedence over positional argument
RESUME=${RESUME:-${7:-""}}

# Training parameters
BATCH_SIZE=${BATCH_SIZE:-1}
GRADIENT_ACCUMULATION=${GRADIENT_ACCUMULATION:-32}
CHUNK_SIZE=${CHUNK_SIZE:-1}
LR=${LR:-2e-5}
ETA=${ETA:-3.0}

# LoRA parameters (set USE_LORA=true to enable)
USE_LORA=${USE_LORA:-false}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-32}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-"q,k,v,o"}

# Distributed training
MASTER_PORT=${MASTER_PORT:-29500}
GPU_COUNT=$(nvidia-smi -L | wc -l)
if [ "$GPU_COUNT" -eq 0 ]; then
    echo "WARNING: No GPUs detected. Falling back to CPU training."
    GPU_COUNT=1
fi

# --- Validate Paths ---
echo "=========================================="
echo "Vidarc Stage 2 Training"
echo "=========================================="
echo "Config: $CONFIG"
echo "Data directory: $DATA_DIR"
echo "Checkpoint directory: $CKPT_DIR"
echo "Stage 1 weights: $PT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "GPUs: $GPU_COUNT"
echo "Batch size: $BATCH_SIZE x $GRADIENT_ACCUMULATION (effective: $((BATCH_SIZE * GRADIENT_ACCUMULATION * GPU_COUNT)))"
echo "Max steps: $MAX_STEPS"
if [ -n "$RESUME" ]; then
    echo "Resume checkpoint: $RESUME"
fi
echo "=========================================="

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "ERROR: Config file not found: $CONFIG"
    echo "Available configs:"
    ls -la configs/*.yaml 2>/dev/null || echo "  No config files found in configs/"
    exit 1
fi

# Read resume path from config if not provided via command line or environment variable
# Priority: RESUME env/arg > config file
# Note: This runs after conda activation, so Python should be available
if [ -z "$RESUME" ]; then
    # Try to extract resume path from config file using Python
    # Look for "resume:" in output section
    # Use python from conda environment (should be in PATH after activation)
    CONFIG_RESUME=$(python -c "
import yaml
import sys
try:
    with open('$CONFIG', 'r') as f:
        config = yaml.safe_load(f)
    if config and 'output' in config and config['output'] and 'resume' in config['output']:
        resume_val = config['output']['resume']
        if resume_val and resume_val != 'null' and resume_val != 'None' and str(resume_val).strip():
            print(str(resume_val).strip())
except Exception as e:
    pass
" 2>/dev/null)
    
    if [ -n "$CONFIG_RESUME" ]; then
        RESUME="$CONFIG_RESUME"
        echo "Found resume path in config: $RESUME"
    fi
fi

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

# Check if resume checkpoint exists (if provided)
RESUME_FLAG=""
if [ -n "$RESUME" ]; then
    if [ ! -f "$RESUME" ]; then
        echo "ERROR: Resume checkpoint not found: $RESUME"
        echo "Please provide a valid checkpoint path or remove the RESUME parameter."
        exit 1
    else
        RESUME_FLAG="--resume $RESUME"
        echo "Resuming from checkpoint: $RESUME"
    fi
fi

# Build LoRA flags
LORA_FLAGS=""
if [ "$USE_LORA" = "true" ]; then
    LORA_FLAGS="--lora --lora-rank $LORA_RANK --lora-alpha $LORA_ALPHA --lora-dropout $LORA_DROPOUT --lora-target-modules $LORA_TARGET_MODULES"
    echo "LoRA enabled: rank=$LORA_RANK, alpha=$LORA_ALPHA, target_modules=$LORA_TARGET_MODULES"
fi

# Build Stochastic Truncation flags
TRUNCATION_FLAGS=""
if [ "$STOCHASTIC_TRUNCATION" = "true" ]; then
    TRUNCATION_FLAGS="--stochastic-truncation"
    echo "Stochastic gradient truncation: ENABLED"
    if [ -n "$TRUNCATION_STRATEGY" ]; then
        TRUNCATION_FLAGS="$TRUNCATION_FLAGS --truncation-strategy $TRUNCATION_STRATEGY"
        echo "  Strategy: $TRUNCATION_STRATEGY"
    fi
    if [ -n "$NUM_INFERENCE_STEPS" ]; then
        TRUNCATION_FLAGS="$TRUNCATION_FLAGS --num-inference-steps $NUM_INFERENCE_STEPS"
        echo "  Inference steps: $NUM_INFERENCE_STEPS"
    fi
elif [ "$STOCHASTIC_TRUNCATION" = "false" ]; then
    TRUNCATION_FLAGS="--no-stochastic-truncation"
    echo "Stochastic gradient truncation: DISABLED"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Generate timestamp for log file
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
LOG_FILE="$SCRIPT_DIR/logs/training_output-${TIMESTAMP}.log"

# Create symlink to latest log file for convenience
LATEST_LOG_LINK="$SCRIPT_DIR/training_output.log"

# --- Launch Training ---
cd "$SCRIPT_DIR"

echo ""
echo "Starting training..."
echo "Log file: $LOG_FILE"
echo ""

if [ "$GPU_COUNT" -eq 1 ]; then
    # Single GPU: run directly without torchrun
    echo "Running on single GPU (no torchrun)"
    python scripts/train_vidarc.py \
        --config "$CONFIG" \
        --data-dir "$DATA_DIR" \
        --ckpt-dir "$CKPT_DIR" \
        $PT_FLAG \
        $RESUME_FLAG \
        $LORA_FLAGS \
        $TRUNCATION_FLAGS \
        --output-dir "$OUTPUT_DIR" \
        --max-steps "$MAX_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --gradient-accumulation "$GRADIENT_ACCUMULATION" \
        --chunk-size "$CHUNK_SIZE" \
        --lr "$LR" \
        --eta "$ETA" \
        --log-interval 10 \
        --save-interval 500 2>&1 | tee "$LOG_FILE"
    
    # Create symlink to latest log
    ln -sf "$LOG_FILE" "$LATEST_LOG_LINK"
    echo "Latest log symlink: $LATEST_LOG_LINK -> $LOG_FILE"
else
    # Multi-GPU: use torchrun
    echo "Running on $GPU_COUNT GPUs with torchrun"
    torchrun --nproc_per_node=$GPU_COUNT --master_port=$MASTER_PORT \
        scripts/train_vidarc.py \
        --config "$CONFIG" \
        --data-dir "$DATA_DIR" \
        --ckpt-dir "$CKPT_DIR" \
        $PT_FLAG \
        $RESUME_FLAG \
        $LORA_FLAGS \
        $TRUNCATION_FLAGS \
        --output-dir "$OUTPUT_DIR" \
        --max-steps "$MAX_STEPS" \
        --batch-size "$BATCH_SIZE" \
        --gradient-accumulation "$GRADIENT_ACCUMULATION" \
        --chunk-size "$CHUNK_SIZE" \
        --lr "$LR" \
        --eta "$ETA" \
        --log-interval 10 \
        --save-interval 500 2>&1 | tee "$LOG_FILE"
    
    # Create symlink to latest log
    ln -sf "$LOG_FILE" "$LATEST_LOG_LINK"
    echo "Latest log symlink: $LATEST_LOG_LINK -> $LOG_FILE"
fi

echo "=========================================="
echo "Training finished."
echo "Output saved to: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "=========================================="

# =============================================================================
# Post-Training Evaluation (Optional)
# =============================================================================
# Set RUN_EVAL=true to automatically run evaluation after training
# Or run manually: TASK_NAME=<task> bash run_eval_ddp_causal.sh ...

RUN_EVAL=${RUN_EVAL:-false}
EVAL_TASK_NAME=${EVAL_TASK_NAME:-"adjust_bottle"}
EVAL_TASK_CONFIG=${EVAL_TASK_CONFIG:-"hd_clean"}

# Evaluation parameters
EVAL_NUM_NEW_FRAMES=${EVAL_NUM_NEW_FRAMES:-16}
EVAL_NUM_SAMPLING_STEP=${EVAL_NUM_SAMPLING_STEP:-10}
EVAL_CFG=${EVAL_CFG:-3.0}

# Model paths for evaluation
EVAL_CKPT=${EVAL_CKPT:-"./checkpoints/vidar/vidarc_4x.pt"}
EVAL_IDM=${EVAL_IDM:-"/mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/vidar/vidar_ckpts/idm.pt"}
EVAL_PREFIX=${EVAL_PREFIX:-"trained_${MAX_STEPS}"}

if [ "$RUN_EVAL" = "true" ]; then
    echo ""
    echo "=========================================="
    echo "Starting Post-Training Evaluation"
    echo "=========================================="
    echo "Task: $EVAL_TASK_NAME"
    echo "Config: $EVAL_TASK_CONFIG"
    echo "Checkpoint: $EVAL_CKPT"
    echo "IDM: $EVAL_IDM"
    echo "Prefix: $EVAL_PREFIX"
    echo "Num new frames: $EVAL_NUM_NEW_FRAMES"
    echo "Sampling steps: $EVAL_NUM_SAMPLING_STEP"
    echo "CFG scale: $EVAL_CFG"
    echo "=========================================="

    # Check if checkpoint exists
    if [ ! -f "$EVAL_CKPT" ]; then
        echo "ERROR: Trained checkpoint not found: $EVAL_CKPT"
        echo "Skipping evaluation."
    else
        # Run evaluation
        TASK_NAME="$EVAL_TASK_NAME" bash run_eval_ddp_causal.sh \
            "$EVAL_TASK_CONFIG" \
            "$EVAL_CKPT" \
            "$EVAL_IDM" \
            "$EVAL_PREFIX" \
            "$EVAL_NUM_NEW_FRAMES" \
            "$EVAL_NUM_SAMPLING_STEP" \
            "$EVAL_CFG"

        EVAL_EXIT_CODE=$?
        if [ $EVAL_EXIT_CODE -eq 0 ]; then
            echo "=========================================="
            echo "Evaluation completed successfully!"
            echo "=========================================="
        else
            echo "=========================================="
            echo "Evaluation failed with exit code: $EVAL_EXIT_CODE"
            echo "=========================================="
        fi
    fi
else
    echo ""
    echo "=========================================="
    echo "To run evaluation manually:"
    echo "=========================================="
    echo "TASK_NAME=$EVAL_TASK_NAME bash run_eval_ddp_causal.sh \\"
    echo "    $EVAL_TASK_CONFIG \\"
    echo "    $EVAL_CKPT \\"
    echo "    $EVAL_IDM \\"
    echo "    trained_${MAX_STEPS} \\"
    echo "    $EVAL_NUM_NEW_FRAMES \\"
    echo "    $EVAL_NUM_SAMPLING_STEP \\"
    echo "    $EVAL_CFG"
    echo "=========================================="
fi
