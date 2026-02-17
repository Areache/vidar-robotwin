#!/usr/bin/env python
"""
WanTI2V Plan Video Quality Verification.

Generates 121-frame plan videos from robot observations using WanTI2V (non-causal)
or WanTI2VCausal (causal) at 3-view unified image resolution (720×640, matching
vidar.pt / vidarc.pt training). Saves MP4 + keyframe strips for visual inspection.

Usage:
    # Quick test: 1 episode, 1 config (non-causal, vidar.pt)
    python experiments/subgoal/verify_plan_quality.py \
        --hdf5 /path/to/episode_000000.hdf5

    # Causal model (vidarc.pt)
    python experiments/subgoal/verify_plan_quality.py \
        --hdf5 /path/to/episode_000000.hdf5 \
        --causal --pt-dir /path/to/vidarc.pt

    # Full test: sweep guide_scale and seeds
    python experiments/subgoal/verify_plan_quality.py \
        --hdf5 /path/to/episode_000000.hdf5 --mode full

    # Custom frame count / resolution
    python experiments/subgoal/verify_plan_quality.py \
        --hdf5 /path/to/episode_000000.hdf5 \
        --frame-num 49 --max-area 471040
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Tuple

import cv2
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # vidar-robotwin
VIDAR_PATH = PROJECT_ROOT.parent / "vidar"
sys.path.insert(0, str(VIDAR_PATH))
sys.path.insert(0, str(PROJECT_ROOT))

import h5py
import torch

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = "/mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/vidar/Wan2.2-TI2V-5B"
DEFAULT_DATA_DIR = "/mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/datasets/robotwin/processed"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "plan_quality"

# Eval resolution: matching official vidar --size "640*736"
PLAN_MAX_AREA = 640 * 736  # 471,040 (official vidar size config)
PLAN_FRAME_NUM = 121       # 4*30+1, ~12s @10fps

# Prompt template — 3-view aloha (matching vidar.pt training / ar.py:set_instruction)
PROMPT_TEMPLATE = (
    "The whole scene is in a realistic, industrial art style with three views: "
    "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
    "The aloha robot is currently performing the following task: {task_description}"
)


def load_first_frame(hdf5_path: str) -> np.ndarray:
    """Load first frame. Prefers unified_image (3-view, matching vidar.pt training). Returns RGB (H, W, 3) uint8."""
    with h5py.File(hdf5_path, "r") as f:
        if "observations/unified_image" in f:
            return f["observations/unified_image"][0]
        if "observations/images/cam_high" in f:
            return f["observations/images/cam_high"][0]
    raise ValueError(f"No image data found in {hdf5_path}")


def load_instruction(hdf5_path: str) -> str:
    """Load task instruction from HDF5 attributes."""
    with h5py.File(hdf5_path, "r") as f:
        if "instruction" in f.attrs:
            instr = f.attrs["instruction"]
            return instr.decode("utf-8") if isinstance(instr, bytes) else str(instr)
    return ""


def load_gt_video(hdf5_path: str, max_frames: int = 121) -> np.ndarray:
    """Load GT video. Prefers unified_image (3-view, matching vidar.pt training). Returns RGB (T, H, W, 3) uint8."""
    with h5py.File(hdf5_path, "r") as f:
        if "observations/unified_image" in f:
            data = f["observations/unified_image"]
        elif "observations/images/cam_high" in f:
            data = f["observations/images/cam_high"]
        else:
            return None
        T = min(data.shape[0], max_frames)
        return data[:T]


def generate_plan_video(
    model,
    first_frame: np.ndarray,
    prompt: str,
    frame_num: int = PLAN_FRAME_NUM,
    max_area: int = PLAN_MAX_AREA,
    sampling_steps: int = 50,
    guide_scale: float = 5.0,
    shift: float = 5.0,
    seed: int = 42,
) -> Tuple[np.ndarray, float]:
    """Generate plan video. Returns (T, H, W, 3) uint8 RGB + elapsed time."""
    pil_image = Image.fromarray(first_frame)

    t0 = time.time()
    video_tensor = model.generate(
        input_prompt=prompt,
        img=pil_image,
        max_area=max_area,
        frame_num=frame_num,
        shift=shift,
        sampling_steps=sampling_steps,
        guide_scale=guide_scale,
        seed=seed,
        offload_model=True,
    )
    elapsed = time.time() - t0

    if video_tensor is None:
        raise RuntimeError("Video generation failed — model returned None")

    # (C, T, H, W) -> (T, H, W, C), [-1,1] -> [0,255]
    video = video_tensor.permute(1, 2, 3, 0).cpu().numpy()
    video = ((video + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return video, elapsed


def save_mp4(video: np.ndarray, path: Path, fps: int = 10):
    """Save video as MP4 using imageio + ffmpeg (yuv420p for compatibility)."""
    import imageio.v3 as iio
    path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, C = video.shape
    # Ensure even dimensions for yuv420p
    H_out = H if H % 2 == 0 else H - 1
    W_out = W if W % 2 == 0 else W - 1
    frames = video[:, :H_out, :W_out, :]
    iio.imwrite(
        str(path),
        frames,
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    print(f"  Saved MP4: {path} ({T} frames, {W_out}x{H_out}, {fps}fps)")


def save_frame_strip(
    video: np.ndarray,
    path: Path,
    num_samples: int = 12,
    label: str = "",
):
    """Save evenly-spaced frames as a horizontal strip image."""
    T = len(video)
    indices = np.linspace(0, T - 1, num_samples, dtype=int)
    frames = [video[i] for i in indices]

    # Add frame index label to each frame
    labeled = []
    for idx, frame in zip(indices, frames):
        f = frame.copy()
        cv2.putText(f, f"f{idx}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        labeled.append(f)

    strip = np.concatenate(labeled, axis=1)  # horizontal concat

    # Add label text at top-left
    if label:
        cv2.putText(strip, label, (5, strip.shape[0] - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f"  Saved strip: {path}")


def save_comparison(
    gt_video: np.ndarray,
    plan_video: np.ndarray,
    path: Path,
    num_samples: int = 10,
):
    """Save GT vs Plan comparison: two rows of frames at matching time indices."""
    T_gt, T_plan = len(gt_video), len(plan_video)

    # Sample matching time points
    indices_gt = np.linspace(0, T_gt - 1, num_samples, dtype=int)
    indices_plan = np.linspace(0, T_plan - 1, num_samples, dtype=int)

    # Resize plan frames to match GT spatial dims for visual comparison
    H_gt, W_gt = gt_video.shape[1:3]
    H_plan, W_plan = plan_video.shape[1:3]

    gt_frames = []
    plan_frames = []
    for ig, ip in zip(indices_gt, indices_plan):
        gf = gt_video[ig].copy()
        pf = plan_video[ip].copy()
        if (H_plan, W_plan) != (H_gt, W_gt):
            pf = cv2.resize(pf, (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)
        cv2.putText(gf, f"GT f{ig}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(pf, f"Plan f{ip}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        gt_frames.append(gf)
        plan_frames.append(pf)

    row_gt = np.concatenate(gt_frames, axis=1)
    row_plan = np.concatenate(plan_frames, axis=1)
    combined = np.concatenate([row_gt, row_plan], axis=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"  Saved comparison: {path}")


def main():
    parser = argparse.ArgumentParser(description="Verify WanTI2V plan video quality")
    parser.add_argument("--hdf5", type=str, required=True, help="Path to HDF5 episode file")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--pt-dir", type=str, default=None,
                        help="Path to fine-tuned weights (e.g. vidar.pt) to load on top of base model")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                        help="quick: 1 config; full: sweep guide_scale × seeds")
    parser.add_argument("--frame-num", type=int, default=PLAN_FRAME_NUM,
                        help="Number of frames to generate (must be 4n+1)")
    parser.add_argument("--max-area", type=int, default=PLAN_MAX_AREA,
                        help="Max pixel area for planning resolution")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of denoising steps (config default: 50)")
    parser.add_argument("--shift", type=float, default=5.0,
                        help="Noise schedule shift (5.0 for 720p, 3.0 for 480p)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Override task prompt (default: read from HDF5)")
    parser.add_argument("--causal", action="store_true",
                        help="Use causal model (WanTI2VCausal) for vidarc.pt checkpoints")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--dry-run", action="store_true", help="Verify paths without loading model")
    args = parser.parse_args()

    # Validate frame_num
    assert (args.frame_num - 1) % 4 == 0, f"frame_num must be 4n+1, got {args.frame_num}"

    # Load episode data
    hdf5_path = args.hdf5
    print(f"Loading episode: {hdf5_path}")
    first_frame = load_first_frame(hdf5_path)
    instruction = load_instruction(hdf5_path)
    gt_video = load_gt_video(hdf5_path, max_frames=args.frame_num)
    episode_name = Path(hdf5_path).stem

    print(f"  First frame shape: {first_frame.shape}")
    print(f"  Instruction: {instruction}")
    if gt_video is not None:
        print(f"  GT video: {gt_video.shape}")

    # Build prompt — use full aloha 3-view template (matching vidar.pt training)
    if args.prompt:
        task_description = args.prompt
    elif instruction:
        # Extract pure task description from full prompt (strip robot/camera prefix)
        if "performing the following task: " in instruction:
            task_description = instruction.split("performing the following task: ")[-1].strip()
        else:
            task_description = instruction
        # Convert task IDs like "stack_bowl_two" → "stack bowl two"
        if "_" in task_description and " " not in task_description:
            task_description = task_description.replace("_", " ")
    else:
        task_description = "stacks the two bowls on top of each other"
        print(f"  WARNING: No instruction found, using default: {task_description}")

    if task_description:
        task_description = task_description[0].lower() + task_description[1:]
    prompt = PROMPT_TEMPLATE.format(task_description=task_description)
    print(f"  Prompt: {prompt[:100]}...")

    # Resolution info
    latent_T = (args.frame_num - 1) // 4 + 1
    print(f"\n  Plan config: frame_num={args.frame_num}, max_area={args.max_area}, "
          f"steps={args.steps}")
    print(f"  Expected latent: T={latent_T} temporal frames")

    if args.dry_run:
        print("\n[DRY RUN] All paths verified. Exiting.")
        return

    # Experiment configs
    if args.mode == "quick":
        configs = [{"guide_scale": 5.0, "seed": 42}]
    else:  # full
        configs = [
            {"guide_scale": gs, "seed": s}
            for gs in [3.0, 5.0, 7.0]
            for s in [42, 123, 456]
        ]

    # Load model
    from wan.configs.wan_ti2v_5B import wan_ti2v_5B

    if args.causal:
        from wan.textimage2video_causal import WanTI2VCausal
        print(f"\nLoading WanTI2VCausal (causal) from {args.checkpoint}...")
        if args.pt_dir:
            print(f"  Fine-tuned weights: {args.pt_dir}")
        model = WanTI2VCausal(
            config=wan_ti2v_5B,
            checkpoint_dir=args.checkpoint,
            pt_dir=args.pt_dir,
            device_id=args.device,
            t5_cpu=True,
            init_on_cpu=True,
            convert_model_dtype=True,
        )
    else:
        from wan.textimage2video import WanTI2V
        print(f"\nLoading WanTI2V (non-causal) from {args.checkpoint}...")
        if args.pt_dir:
            print(f"  Fine-tuned weights: {args.pt_dir}")
        model = WanTI2V(
            config=wan_ti2v_5B,
            checkpoint_dir=args.checkpoint,
            pt_dir=args.pt_dir,
            device_id=args.device,
            t5_cpu=True,
            init_on_cpu=True,
            convert_model_dtype=True,
        )
    print("Model loaded.\n")

    # Output directory
    out_dir = Path(args.output_dir) / episode_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save GT strip for reference
    if gt_video is not None:
        save_frame_strip(gt_video, out_dir / "gt_strip.jpg", label="GT")
        save_mp4(gt_video, out_dir / "gt_video.mp4")

    # Save first frame
    cv2.imwrite(str(out_dir / "obs_frame0.jpg"),
                cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    # Generate plan videos
    for i, cfg in enumerate(configs):
        tag = f"cfg{cfg['guide_scale']}_seed{cfg['seed']}"
        print(f"\n[{i+1}/{len(configs)}] Generating plan: {tag}")
        print(f"  frame_num={args.frame_num}, max_area={args.max_area}, "
              f"steps={args.steps}, shift={args.shift}, guide_scale={cfg['guide_scale']}, seed={cfg['seed']}")

        plan_video, elapsed = generate_plan_video(
            model,
            first_frame,
            prompt,
            frame_num=args.frame_num,
            max_area=args.max_area,
            sampling_steps=args.steps,
            guide_scale=cfg["guide_scale"],
            shift=args.shift,
            seed=cfg["seed"],
        )
        print(f"  Generated: {plan_video.shape}, elapsed={elapsed:.1f}s")

        # Save outputs
        save_mp4(plan_video, out_dir / f"plan_{tag}.mp4")
        save_frame_strip(plan_video, out_dir / f"plan_strip_{tag}.jpg",
                         label=f"Plan {tag} ({elapsed:.1f}s)")
        if gt_video is not None:
            save_comparison(gt_video, plan_video,
                            out_dir / f"comparison_{tag}.jpg")

    print(f"\nDone! Results saved to: {out_dir}")
    print(f"Files:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
