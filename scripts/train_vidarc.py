#!/mnt/shared-storage-user/qinyiran/cyujie/cyujie/env/RoboTwin-hb/bin/python
"""
Train Vidarc (Stage 2): Causal fine-tuning with Self-Forcing.

This script implements Stage 2 training from the Vidarc paper:
- Starts from Stage 1 fine-tuned weights (vidar.pt)
- Uses causal attention with KV caching
- Self-Forcing training paradigm
- Optional embodiment-aware loss

Example usage:
    # Recommended: Use the shell wrapper (consistent with run_eval_ddp_causal.sh)
    bash run_train_vidarc.sh \
        configs/vidarc_2xh200.yaml \
        /path/to/dataset \
        /path/to/Wan2.2-TI2V-5B \
        /path/to/vidar.pt \
        ./output_vidarc \
        4000

    # Alternative: Manual activation (same pattern as run_eval_ddp_causal.sh)
    conda activate /mnt/shared-storage-user/qinyiran/cyujie/cyujie/env/RoboTwin-hb
    torchrun --nproc_per_node=2 scripts/train_vidarc.py \
        --config configs/vidarc_2xh200.yaml \
        --data-dir /path/to/dataset \
        --ckpt-dir /path/to/Wan2.2-TI2V-5B \
        --pt-dir /path/to/vidar.pt \
        --max-steps 4000

    # Direct Python call (not recommended for distributed training)
    /mnt/shared-storage-user/qinyiran/cyujie/cyujie/env/RoboTwin-hb/bin/python scripts/train_vidarc.py \
        --config configs/vidarc_causal.yaml \
        --data-dir /path/to/dataset \
        --ckpt-dir /path/to/Wan2.2-TI2V-5B \
        --pt-dir /path/to/vidar.pt 
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Initialize logging early (before any error handling that might use logger)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Ensure we're using the correct Python environment
# This matches the pattern used in run_eval_ddp_causal.sh and run_eval_ddp.py
TARGET_PYTHON = "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/env/RoboTwin-hb/bin/python"
TARGET_ENV_PATH = "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/env/RoboTwin-hb"

# Check if running under torchrun (distributed training)
is_distributed = "LOCAL_RANK" in os.environ or "RANK" in os.environ

if sys.executable != TARGET_PYTHON:
    # When using torchrun, verify the environment is correct
    if is_distributed:
        # Ensure PATH includes the conda environment bin (for finding tools like ffmpeg)
        conda_env_bin = f"{TARGET_ENV_PATH}/bin"
        current_path = os.environ.get("PATH", "")
        if conda_env_bin not in current_path:
            os.environ["PATH"] = f"{conda_env_bin}:{current_path}"
        
        # Try to import critical dependencies to verify environment
        try:
            import h5py
        except ImportError:
            error_msg = (
                f"Error: h5py not found in current Python ({sys.executable}).\n"
                f"Expected Python: {TARGET_PYTHON}\n"
                f"Please use the shell wrapper script:\n"
                f"  bash run_train_vidarc.sh\n"
                f"Or activate the conda environment first:\n"
                f"  conda activate {TARGET_ENV_PATH}\n"
                f"Then run: torchrun --nproc_per_node=N scripts/train_vidarc.py ..."
            )
            logger.error(error_msg)
            print(error_msg, file=sys.stderr)
            raise

# Add training module to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.distributed as dist

from training.config import load_config, VidarConfig
from training.trainers.vidarc_trainer import create_vidarc_trainer
from training.distributed.fsdp_utils import setup_distributed, cleanup_distributed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train Vidarc (Stage 2): Causal fine-tuning with Self-Forcing"
    )

    # Config
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to config YAML file"
    )

    # Paths
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to training data directory"
    )
    parser.add_argument(
        "--ckpt-dir", type=str, required=True,
        help="Path to Wan2.2-TI2V-5B checkpoint directory"
    )
    parser.add_argument(
        "--pt-dir", type=str, default=None,
        help="Path to Stage 1 weights (vidar.pt) - REQUIRED for Stage 2"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./output_vidarc",
        help="Output directory for checkpoints and logs"
    )

    # Training overrides
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Maximum training steps (default: from config)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Per-GPU batch size (default: from config)"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: from config)"
    )
    parser.add_argument(
        "--gradient-accumulation", type=int, default=None,
        help="Gradient accumulation steps"
    )

    # Self-Forcing parameters
    parser.add_argument(
        "--chunk-size", type=int, default=16,
        help="Chunk size for Self-Forcing training (default: 16)"
    )
    parser.add_argument(
        "--eta", type=float, default=3.0,
        help="Embodiment loss weight (default: 3.0)"
    )
    parser.add_argument(
        "--use-embodiment-loss", action="store_true",
        help="Enable embodiment-aware loss weighting"
    )

    # Resume
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from"
    )

    # Logging
    parser.add_argument(
        "--log-interval", type=int, default=10,
        help="Steps between log messages"
    )
    parser.add_argument(
        "--save-interval", type=int, default=500,
        help="Steps between checkpoint saves"
    )

    # Debug
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug mode (more verbose logging)"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup distributed if using torchrun
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_main = local_rank == 0

    if is_main:
        logger.info("=" * 60)
        logger.info("Vidarc Stage 2 Training: Causal Self-Forcing")
        logger.info("=" * 60)
        logger.info(f"World size: {world_size}")
        logger.info(f"Device: {device}")

    # Load config
    config = load_config(args.config)

    # Override with command line args
    config.data.data_dir = args.data_dir
    config.model.ckpt_dir = args.ckpt_dir
    config.model.pt_dir = args.pt_dir
    config.output.output_dir = args.output_dir

    # Stage 2 specific settings
    config.model.chunk_size = args.chunk_size
    config.training.eta = args.eta
    config.training.use_embodiment_loss = args.use_embodiment_loss

    if args.max_steps is not None:
        config.training.max_steps = args.max_steps
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.lr = args.lr
    if args.gradient_accumulation is not None:
        config.training.gradient_accumulation_steps = args.gradient_accumulation

    config.output.log_interval = args.log_interval
    config.output.save_interval = args.save_interval

    # Warn if no Stage 1 weights provided
    if args.pt_dir is None and is_main:
        logger.warning("=" * 60)
        logger.warning("WARNING: No Stage 1 weights (--pt-dir) provided!")
        logger.warning("Stage 2 training typically starts from Stage 1 checkpoint.")
        logger.warning("Training will start from base Wan2.2 weights.")
        logger.warning("=" * 60)

    # Create output directory
    output_dir = Path(args.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save config
        config_save_path = output_dir / "config.yaml"
        config.save(str(config_save_path))
        logger.info(f"Saved config to {config_save_path}")

    # Create trainer
    if is_main:
        logger.info("Creating Vidarc causal trainer...")

    trainer = create_vidarc_trainer(config)

    # Setup
    trainer.setup()

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Log training info
    if is_main:
        logger.info("=" * 60)
        logger.info("Stage 2 Training Configuration:")
        logger.info(f"  - Data dir: {args.data_dir}")
        logger.info(f"  - Checkpoint dir: {args.ckpt_dir}")
        logger.info(f"  - Stage 1 weights: {args.pt_dir}")
        logger.info(f"  - Output dir: {args.output_dir}")
        logger.info(f"  - Max steps: {config.training.max_steps}")
        logger.info(f"  - Batch size: {config.training.batch_size} per GPU")
        logger.info(f"  - Learning rate: {config.training.lr}")
        logger.info(f"  - Chunk size: {args.chunk_size}")
        logger.info(f"  - Embodiment loss: {args.use_embodiment_loss} (eta={args.eta})")
        logger.info("=" * 60)

    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        if is_main:
            logger.info("Training interrupted by user")
            # Save final checkpoint
            trainer.save_checkpoint(str(output_dir / "vidarc_interrupted.pt"))
    finally:
        if world_size > 1:
            cleanup_distributed()

    if is_main:
        logger.info("Training complete!")
        logger.info(f"Final checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()
