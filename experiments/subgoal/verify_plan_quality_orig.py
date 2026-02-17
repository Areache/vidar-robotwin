#!/usr/bin/env python
"""
Isolation test: Use ORIGINAL Wan2.2 code (not vidar fork) to generate
robot scene video from the same observation image.

This helps determine whether generation failure is:
  A) Code-level issue in vidar fork  → original works, vidar doesn't
  B) Model-level issue               → both fail on robot scenes

Usage:
    python experiments/subgoal/verify_plan_quality_orig.py \
        --hdf5 /path/to/episode_000000.hdf5 \
        --checkpoint /path/to/Wan2.2-TI2V-5B

    # Or with a pre-saved PNG instead of HDF5:
    python experiments/subgoal/verify_plan_quality_orig.py \
        --image /path/to/obs.png \
        --checkpoint /path/to/Wan2.2-TI2V-5B
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Use ORIGINAL Wan2.2 code — must be imported BEFORE any vidar code
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
ORIG_WAN_DIR = SCRIPT_DIR / "Wan2.2"
assert ORIG_WAN_DIR.exists(), f"Original Wan2.2 not found at {ORIG_WAN_DIR}. Run: git clone https://github.com/Wan-Video/Wan2.2.git {ORIG_WAN_DIR}"

# Prepend original Wan2.2 to sys.path so `import wan` resolves to original code
sys.path.insert(0, str(ORIG_WAN_DIR))

import torch
from PIL import Image

# These imports come from the ORIGINAL Wan2.2 (not vidar fork)
from wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.textimage2video import WanTI2V
from wan.utils.utils import save_video

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = "/mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/vidar/Wan2.2-TI2V-5B"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "plan_quality_orig"

PLAN_FRAME_NUM = 121       # 4*30+1, ~12s @10fps
PLAN_MAX_AREA = 320 * 384  # 122,880

PROMPT_TEMPLATE = (
    "A top-down view of an aloha robot on a table. "
    "The aloha robot {task_description}"
)


def load_first_frame_from_hdf5(hdf5_path: str, camera: str = "cam_high") -> np.ndarray:
    """Load first frame from HDF5. Returns RGB (H, W, 3) uint8."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        key = f"observations/images/{camera}"
        if key in f:
            return f[key][0]
        if "observations/unified_image" in f:
            return f["observations/unified_image"][0]
    raise ValueError(f"No image data found in {hdf5_path}")


def load_instruction_from_hdf5(hdf5_path: str) -> str:
    """Load task instruction from HDF5 attributes."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        if "instruction" in f.attrs:
            instr = f.attrs["instruction"]
            return instr.decode("utf-8") if isinstance(instr, bytes) else str(instr)
    return ""


def load_gt_video_from_hdf5(hdf5_path: str, max_frames: int = 121, camera: str = "cam_high") -> np.ndarray:
    """Load GT video for comparison. Returns RGB (T, H, W, 3) uint8."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        key = f"observations/images/{camera}"
        if key in f:
            data = f[key]
        elif "observations/unified_image" in f:
            data = f["observations/unified_image"]
        else:
            return None
        T = min(data.shape[0], max_frames)
        return data[:T]


def save_mp4(video: np.ndarray, path: Path, fps: int = 10):
    """Save video as MP4 using imageio + ffmpeg (yuv420p)."""
    import imageio.v3 as iio
    path.parent.mkdir(parents=True, exist_ok=True)
    T, H, W, C = video.shape
    H_out = H if H % 2 == 0 else H - 1
    W_out = W if W % 2 == 0 else W - 1
    frames = video[:, :H_out, :W_out, :]
    iio.imwrite(
        str(path), frames, fps=fps,
        codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"],
    )
    print(f"  Saved MP4: {path} ({T} frames, {W_out}x{H_out}, {fps}fps)")


def save_comparison_strip(
    video: np.ndarray, path: Path, num_samples: int = 12, label: str = ""
):
    """Save evenly-spaced frames as horizontal strip."""
    import cv2
    T = len(video)
    indices = np.linspace(0, T - 1, num_samples, dtype=int)
    frames = []
    for idx in indices:
        f = video[idx].copy()
        cv2.putText(f, f"f{idx}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        frames.append(f)
    strip = np.concatenate(frames, axis=1)
    if label:
        cv2.putText(strip, label, (5, strip.shape[0] - 10),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f"  Saved strip: {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Isolation test: generate robot video with ORIGINAL Wan2.2 code")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--hdf5", type=str, help="Path to HDF5 episode file")
    group.add_argument("--image", type=str, help="Path to observation PNG/JPG")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--frame-num", type=int, default=PLAN_FRAME_NUM)
    parser.add_argument("--max-area", type=int, default=PLAN_MAX_AREA)
    parser.add_argument("--steps", type=int, default=50,
                        help="Denoising steps (original default=50)")
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0,
                        help="Noise schedule shift (5.0 for 720p, 3.0 for 480p)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    assert (args.frame_num - 1) % 4 == 0, f"frame_num must be 4n+1, got {args.frame_num}"

    # --- Load input image ---
    gt_video = None
    if args.hdf5:
        print(f"Loading from HDF5: {args.hdf5}")
        first_frame = load_first_frame_from_hdf5(args.hdf5)
        instruction = load_instruction_from_hdf5(args.hdf5)
        gt_video = load_gt_video_from_hdf5(args.hdf5, max_frames=args.frame_num)
        episode_name = Path(args.hdf5).stem
    else:
        print(f"Loading image: {args.image}")
        first_frame = np.array(Image.open(args.image).convert("RGB"))
        instruction = ""
        episode_name = Path(args.image).stem

    print(f"  Frame shape: {first_frame.shape}")
    print(f"  Instruction: {instruction}")

    # --- Build prompt ---
    if args.prompt:
        task_desc = args.prompt
    elif instruction:
        task_desc = instruction
    else:
        task_desc = "stacks the two bowls on top of each other"
        print(f"  WARNING: No instruction, using default: {task_desc}")

    prompt = PROMPT_TEMPLATE.format(task_description=task_desc)
    print(f"  Prompt: {prompt[:120]}...")

    # --- Load model (ORIGINAL Wan2.2 code) ---
    cfg = WAN_CONFIGS["ti2v-5B"]
    print(f"\n[ORIGINAL Wan2.2] Loading WanTI2V from {args.checkpoint}")
    print(f"  Config: dim={cfg.dim}, num_heads={cfg.num_heads}, num_layers={cfg.num_layers}")
    print(f"  sample_neg_prompt present: {hasattr(cfg, 'sample_neg_prompt')}")
    if hasattr(cfg, 'sample_neg_prompt'):
        print(f"  sample_neg_prompt: {cfg.sample_neg_prompt[:80]}...")

    model = WanTI2V(
        config=cfg,
        checkpoint_dir=args.checkpoint,
        device_id=args.device,
        t5_cpu=True,
        init_on_cpu=True,
    )
    print("Model loaded.\n")

    # --- Generate ---
    pil_image = Image.fromarray(first_frame)

    print(f"Generating video: frame_num={args.frame_num}, max_area={args.max_area}, "
          f"steps={args.steps}, guide_scale={args.guide_scale}, shift={args.shift}, seed={args.seed}")

    t0 = time.time()
    video_tensor = model.generate(
        input_prompt=prompt,
        img=pil_image,
        max_area=args.max_area,
        frame_num=args.frame_num,
        shift=args.shift,
        sampling_steps=args.steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
        offload_model=True,
    )
    elapsed = time.time() - t0
    print(f"  Generation done: {elapsed:.1f}s")

    if video_tensor is None:
        print("ERROR: model returned None")
        return

    # (C, T, H, W) -> (T, H, W, C), [-1,1] -> [0,255]
    video = video_tensor.permute(1, 2, 3, 0).cpu().numpy()
    video = ((video + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    print(f"  Output shape: {video.shape}")

    # --- Save outputs ---
    out_dir = Path(args.output_dir) / episode_name
    out_dir.mkdir(parents=True, exist_ok=True)

    import cv2
    cv2.imwrite(str(out_dir / "obs_frame0.jpg"),
                cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    tag = f"orig_cfg{args.guide_scale}_seed{args.seed}_steps{args.steps}"
    save_mp4(video, out_dir / f"plan_{tag}.mp4")
    save_comparison_strip(video, out_dir / f"plan_strip_{tag}.jpg",
                          label=f"ORIGINAL {tag} ({elapsed:.1f}s)")

    if gt_video is not None:
        save_mp4(gt_video, out_dir / "gt_video.mp4")
        save_comparison_strip(gt_video, out_dir / "gt_strip.jpg", label="GT")

    # Also save using Wan2.2's own save_video for sanity check
    save_video(
        tensor=video_tensor[None],
        save_file=str(out_dir / f"plan_{tag}_wansave.mp4"),
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    print(f"  Also saved via Wan2.2's save_video utility")

    print(f"\nDone! Results at: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
