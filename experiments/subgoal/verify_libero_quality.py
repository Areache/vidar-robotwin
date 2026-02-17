#!/usr/bin/env python
"""
LIBERO Video Model Subgoal Quality Verification on RoboTwin.

Loads the LIBERO video model (3D UNet, 128x128, trained on LIBERO sim),
generates subgoals autoregressively from a RoboTwin observation,
and saves visual comparisons against GT demo frames.

Usage:
    # From HDF5 demo (has GT comparison)
    python experiments/subgoal/verify_libero_quality.py \
        --hdf5 /path/to/episode_000000.hdf5

    # From eval result video (no GT, just visualize predictions)
    python experiments/subgoal/verify_libero_quality.py \
        --video /path/to/eval_result/adjust_bottle/episode0.mp4

    # Custom model path / milestone
    python experiments/subgoal/verify_libero_quality.py \
        --hdf5 /path/to/episode_000000.hdf5 \
        --model-path /path/to/libero_ep20_bs12_aug \
        --milestone 180000
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

import cv2
from PIL import Image

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # vidar-robotwin

# Try multiple known locations for the video-to-action codebase
_VM_CANDIDATES = [
    PROJECT_ROOT.parent / "vidar" / "vm",  # relative to project
    Path("/mnt/shared-storage-user/kangli/workspace/cyujie/code/vidar/vm"),
    Path("/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm"),
]
VM_PATH = None
for _p in _VM_CANDIDATES:
    if _p.exists():
        VM_PATH = str(_p)
        break
if VM_PATH is None:
    VM_PATH = str(_VM_CANDIDATES[0])  # fallback, will error later with clear message

sys.path.insert(0, VM_PATH)
sys.path.insert(0, str(PROJECT_ROOT))

import torch

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
_MODEL_CANDIDATES = [
    "/mnt/shared-storage-user/kangli/workspace/cyujie/code/vidar/vm/ckpts/libero/libero_ep20_bs12_aug",
    "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm/ckpts/libero/libero_ep20_bs12_aug",
]
DEFAULT_MODEL_PATH = next((p for p in _MODEL_CANDIDATES if os.path.exists(p)), _MODEL_CANDIDATES[0])
DEFAULT_MILESTONE = 180000
DEFAULT_DATA_DIR = "/mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/datasets/robotwin/processed"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "results" / "libero_quality"

SUBGOAL_INTERVAL = 8
NUM_SUBGOALS = 20


def load_first_frame_hdf5(hdf5_path: str) -> np.ndarray:
    """Load first frame from HDF5. Prefers unified_image (3-view). Returns RGB (H, W, 3) uint8."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        if "observations/unified_image" in f:
            return f["observations/unified_image"][0]
        if "observations/images/cam_high" in f:
            return f["observations/images/cam_high"][0]
    raise ValueError(f"No image data found in {hdf5_path}")


def load_instruction_hdf5(hdf5_path: str) -> str:
    """Load task instruction from HDF5 attributes."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        if "instruction" in f.attrs:
            instr = f.attrs["instruction"]
            return instr.decode("utf-8") if isinstance(instr, bytes) else str(instr)
    return ""


def load_gt_video_hdf5(hdf5_path: str, max_frames: int = 200) -> np.ndarray:
    """Load GT video from HDF5. Prefers unified_image (3-view). Returns RGB (T, H, W, 3) uint8."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        if "observations/unified_image" in f:
            data = f["observations/unified_image"]
        elif "observations/images/cam_high" in f:
            data = f["observations/images/cam_high"]
        else:
            return None
        T = min(data.shape[0], max_frames)
        return data[:T]


def load_first_frame_video(video_path: str) -> np.ndarray:
    """Load first frame from MP4 video. Returns RGB (H, W, 3) uint8."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read video: {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def load_video_frames(video_path: str, max_frames: int = 200) -> np.ndarray:
    """Load all frames from MP4 video. Returns RGB (T, H, W, 3) uint8."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.array(frames) if frames else None


def frame_to_model_input(frame_rgb: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert RGB uint8 frame to LIBERO model input (1, 3, 128, 128) float [0,1]."""
    img = cv2.resize(frame_rgb, (128, 128))
    tensor = torch.from_numpy(img).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 128, 128)
    return tensor.to(device)


def tensor_to_rgb(tensor: torch.Tensor) -> np.ndarray:
    """Convert (1, 3, H, W) or (3, H, W) float tensor to RGB uint8."""
    if tensor.dim() == 4:
        tensor = tensor[0]
    img = (tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return img


def load_libero_model(model_path: str, milestone: int, device_id: int = 0):
    """Load LIBERO video model."""
    print(f"Loading LIBERO video model...")
    print(f"  VM_PATH: {VM_PATH}")
    print(f"  Model path: {model_path}")
    print(f"  Milestone: {milestone}")

    from diffuser.libero.lb_video_model_utils import lb_get_video_model_gcp_v2

    model = lb_get_video_model_gcp_v2(
        ckpts_dir=model_path,
        milestone=milestone,
        flow=False,
    )
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    model.ema.ema_model.var_temp = 1.0
    model.ema.ema_model.is_ddim_sampling = False

    print(f"  Loaded on {device}")
    return model, device


def generate_subgoals(
    model,
    device: torch.device,
    first_frame: np.ndarray,
    instruction: str,
    num_subgoals: int = NUM_SUBGOALS,
    subgoal_interval: int = SUBGOAL_INTERVAL,
) -> list:
    """
    Generate subgoals autoregressively (same logic as vm_subgoal_generator.py).

    Returns list of (frame_rgb_128x128, raw_pred_video_7frames) tuples.
    """
    current_input = frame_to_model_input(first_frame, device)
    # Extract pure task description from full prompt (consistent with vm_subgoal_generator.py)
    if "performing the following task: " in instruction:
        task_str = instruction.split("performing the following task: ")[-1].strip()
    else:
        task_str = instruction
    tasks_str = [task_str]

    results = []
    # First subgoal = observation
    results.append({
        "index": 0,
        "frame": cv2.resize(first_frame, (128, 128)),
        "raw_pred": None,
        "label": "obs (input)",
    })

    for i in range(1, num_subgoals):
        t0 = time.time()
        with torch.no_grad():
            preds = model.forward(current_input, tasks_str)

        elapsed = time.time() - t0

        assert len(preds) == 1
        pred_v = preds[0]  # (T, 3, H, W), T typically = 7

        # Select subgoal frame
        T_pred = pred_v.shape[0]
        sg_idx = min(subgoal_interval - 1, T_pred - 1)
        subgoal_tensor = pred_v[sg_idx]  # (3, H, W)

        # Save all 7 predicted frames for inspection
        raw_frames = []
        for t in range(T_pred):
            raw_frames.append(tensor_to_rgb(pred_v[t]))

        results.append({
            "index": i,
            "frame": tensor_to_rgb(subgoal_tensor),
            "raw_pred": raw_frames,
            "label": f"sg{i} (pred_f{sg_idx}/{T_pred}, {elapsed:.2f}s)",
            "elapsed": elapsed,
        })

        # Update input for next iteration
        current_input = pred_v[sg_idx].unsqueeze(0).to(device)
        current_input = torch.clamp(current_input, 0.0, 1.0)

    return results


def save_subgoal_strip(results: list, path: Path, title: str = ""):
    """Save horizontal strip of all generated subgoals."""
    frames = []
    for r in results:
        f = r["frame"].copy()
        # Label
        cv2.putText(f, r["label"][:20], (2, 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        frames.append(f)

    strip = np.concatenate(frames, axis=1)
    if title:
        # Add title bar
        bar = np.zeros((25, strip.shape[1], 3), dtype=np.uint8)
        cv2.putText(bar, title, (5, 18),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        strip = np.concatenate([bar, strip], axis=0)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))
    print(f"  Saved subgoal strip: {path}")


def save_raw_pred_grid(results: list, path: Path):
    """Save grid of all raw predictions (each row = one model.forward call)."""
    rows = []
    for r in results:
        if r["raw_pred"] is None:
            continue
        # Annotate each frame
        labeled = []
        for t, f in enumerate(r["raw_pred"]):
            fc = f.copy()
            cv2.putText(fc, f"t{t}", (2, 12),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            labeled.append(fc)
        row = np.concatenate(labeled, axis=1)
        # Row label
        label_bar = np.zeros((15, row.shape[1], 3), dtype=np.uint8)
        cv2.putText(label_bar, f"step {r['index']}", (2, 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
        rows.append(np.concatenate([label_bar, row], axis=0))

    if not rows:
        return

    grid = np.concatenate(rows, axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"  Saved raw prediction grid: {path}")


def save_comparison(
    gt_video: np.ndarray,
    results: list,
    path: Path,
    subgoal_interval: int = SUBGOAL_INTERVAL,
):
    """
    Save GT vs Predicted comparison.
    Top row: GT frames at subgoal timestamps.
    Bottom row: LIBERO predicted subgoals.
    """
    T_gt = len(gt_video)
    pairs = []

    for r in results:
        gt_frame_idx = min(r["index"] * subgoal_interval, T_gt - 1)
        gt_frame = cv2.resize(gt_video[gt_frame_idx], (128, 128))
        pred_frame = r["frame"]

        # Label
        gf = gt_frame.copy()
        pf = pred_frame.copy()
        cv2.putText(gf, f"GT f{gt_frame_idx}", (2, 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.putText(pf, f"Pred sg{r['index']}", (2, 12),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        pairs.append((gf, pf))

    row_gt = np.concatenate([p[0] for p in pairs], axis=1)
    row_pred = np.concatenate([p[1] for p in pairs], axis=1)

    # Labels
    bar_gt = np.zeros((20, row_gt.shape[1], 3), dtype=np.uint8)
    bar_pred = np.zeros((20, row_pred.shape[1], 3), dtype=np.uint8)
    cv2.putText(bar_gt, "GT (RoboTwin demo)", (5, 15),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    cv2.putText(bar_pred, "LIBERO model prediction", (5, 15),
                 cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

    combined = np.concatenate([bar_gt, row_gt, bar_pred, row_pred], axis=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"  Saved comparison: {path}")


def main():
    parser = argparse.ArgumentParser(description="Verify LIBERO video model subgoal quality on RoboTwin")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--hdf5", type=str, help="Path to HDF5 demo episode (has GT for comparison)")
    src.add_argument("--video", type=str, help="Path to eval result MP4 (no GT comparison)")

    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="LIBERO video model checkpoint directory")
    parser.add_argument("--milestone", type=int, default=DEFAULT_MILESTONE,
                        help="Checkpoint milestone number")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--num-subgoals", type=int, default=NUM_SUBGOALS,
                        help="Number of subgoals to generate")
    parser.add_argument("--interval", type=int, default=SUBGOAL_INTERVAL,
                        help="Subgoal interval (frames between subgoals)")
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override task instruction text")
    parser.add_argument("--vm-path", type=str, default=None,
                        help="Override path to video-to-action codebase (contains diffuser/)")
    parser.add_argument("--device", type=int, default=0, help="GPU device ID")
    parser.add_argument("--dry-run", action="store_true", help="Verify paths without running model")
    args = parser.parse_args()

    # Override VM_PATH if provided
    global VM_PATH
    if args.vm_path:
        VM_PATH = args.vm_path
        if VM_PATH not in sys.path:
            sys.path.insert(0, VM_PATH)

    # ---- Load source data ----
    if args.hdf5:
        source_path = args.hdf5
        first_frame = load_first_frame_hdf5(source_path)
        instruction = args.instruction or load_instruction_hdf5(source_path)
        gt_video = load_gt_video_hdf5(source_path)
        episode_name = Path(source_path).stem
    else:
        source_path = args.video
        first_frame = load_first_frame_video(source_path)
        instruction = args.instruction or ""
        gt_video = load_video_frames(source_path)
        episode_name = Path(source_path).stem

    print(f"Source: {source_path}")
    print(f"  First frame: {first_frame.shape}")
    print(f"  Instruction: '{instruction}'")
    if gt_video is not None:
        print(f"  GT video: {gt_video.shape}")
    print(f"  Num subgoals: {args.num_subgoals}, interval: {args.interval}")

    if not instruction:
        instruction = "performs a manipulation task"
        print(f"  WARNING: No instruction found, using default: '{instruction}'")

    if args.dry_run:
        print(f"\n  Model path: {args.model_path}")
        print(f"  Milestone: {args.milestone}")
        print(f"  Model path exists: {os.path.exists(args.model_path)}")
        print("\n[DRY RUN] All paths verified. Exiting.")
        return

    # ---- Load model ----
    model, device = load_libero_model(args.model_path, args.milestone, args.device)

    # ---- Generate subgoals ----
    print(f"\nGenerating {args.num_subgoals} subgoals (interval={args.interval})...")
    t0 = time.time()
    results = generate_subgoals(
        model, device, first_frame, instruction,
        num_subgoals=args.num_subgoals,
        subgoal_interval=args.interval,
    )
    total_elapsed = time.time() - t0
    print(f"  Total generation time: {total_elapsed:.1f}s")
    print(f"  Avg per subgoal: {total_elapsed / max(1, args.num_subgoals - 1):.2f}s")

    # ---- Save outputs ----
    out_dir = Path(args.output_dir) / episode_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # First frame at original resolution
    cv2.imwrite(str(out_dir / "obs_frame0.jpg"),
                cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))

    # Subgoal strip
    save_subgoal_strip(results, out_dir / "subgoal_strip.jpg",
                       title=f"LIBERO subgoals | instr: {instruction[:60]}")

    # Raw prediction grid (all 7 frames per step)
    save_raw_pred_grid(results, out_dir / "raw_pred_grid.jpg")

    # GT comparison (if available)
    if gt_video is not None:
        save_comparison(gt_video, results, out_dir / "comparison_gt_vs_pred.jpg",
                        subgoal_interval=args.interval)

    print(f"\nDone! Results saved to: {out_dir}")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
