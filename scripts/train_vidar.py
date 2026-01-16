#!/usr/bin/env python3
"""
Stage 1 Training Script: Vidar fine-tuning.

Usage:
    # Single GPU
    python scripts/train_vidar.py --config configs/vidar_finetune.yaml

    # Multi-GPU with torchrun
    torchrun --nproc_per_node=2 scripts/train_vidar.py --config configs/vidar_finetune.yaml

    # Multi-GPU with SLURM
    srun python scripts/train_vidar.py --config configs/vidar_finetune.yaml
"""

import os
import sys
import argparse
import logging

# Add training module to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from training.config import VidarConfig
from training.trainers import VidarTrainer
from training.distributed import is_main_process

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: Vidar fine-tuning")

    # Config
    parser.add_argument(
        "--config",
        type=str,
        default="configs/vidar_finetune.yaml",
        help="Path to config YAML file",
    )

    # Override options
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Override: Path to Wan2.2-TI2V-5B checkpoint",
    )
    parser.add_argument(
        "--pt-dir",
        type=str,
        default=None,
        help="Override: Path to fine-tuned weights (optional)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Override: Path to training data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override: Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override: Batch size per GPU",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override: Learning rate",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override: Number of training steps",
    )
    parser.add_argument(
        "--gradient-accumulation",
        type=int,
        default=None,
        help="Override: Gradient accumulation steps",
    )

    # Resume
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )

    # Logging
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Override: WandB project name",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable WandB logging",
    )

    # Debug
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (small batch, few steps)",
    )

    return parser.parse_args()


def apply_overrides(config: VidarConfig, args) -> VidarConfig:
    """Apply command line overrides to config."""
    # Model overrides
    if args.ckpt_dir:
        config.model.ckpt_dir = args.ckpt_dir
    if args.pt_dir:
        config.model.pt_dir = args.pt_dir

    # Training overrides
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.lr:
        config.training.lr = args.lr
    if args.num_steps:
        config.training.num_steps = args.num_steps
    if args.gradient_accumulation:
        config.training.gradient_accumulation = args.gradient_accumulation

    # Data overrides
    if args.data_dir:
        config.data.data_dir = args.data_dir

    # Logging overrides
    if args.output_dir:
        config.logging.output_dir = args.output_dir
    if args.wandb_project:
        config.logging.wandb_project = args.wandb_project
    if args.no_wandb:
        config.logging.use_wandb = False

    # Debug mode
    if args.debug:
        config.training.batch_size = 1
        config.training.num_steps = 10
        config.training.gradient_accumulation = 1
        config.logging.log_interval = 1
        config.logging.save_interval = 5
        config.logging.use_wandb = False

    return config


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )


def main():
    args = parse_args()
    setup_logging()

    # Load config
    config = VidarConfig.from_yaml(args.config)
    config = apply_overrides(config, args)

    if is_main_process():
        logger.info("=" * 60)
        logger.info("Stage 1: Vidar Fine-tuning")
        logger.info("=" * 60)
        logger.info(f"Config: {args.config}")
        logger.info(f"Checkpoint: {config.model.ckpt_dir}")
        logger.info(f"Data: {config.data.data_dir}")
        logger.info(f"Output: {config.logging.output_dir}")
        logger.info(f"Batch size: {config.training.batch_size}")
        logger.info(f"Learning rate: {config.training.lr}")
        logger.info(f"Num steps: {config.training.num_steps}")
        logger.info("=" * 60)

    # Create trainer
    trainer = VidarTrainer(config)

    try:
        # Setup
        trainer.setup()

        # Resume if specified
        if args.resume:
            trainer.load_checkpoint(args.resume)

        # Train
        trainer.train()

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
