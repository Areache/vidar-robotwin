"""
Step 4: Build GT Keyframe Library for Method B

Supports two data sources:
  1. HDF5 files — phase boundaries from gripper state changes (action data)
  2. Video files (mp4) — phase boundaries from visual/semantic detection (no action data needed)

The library maps (task, phase_id) -> keyframe images, enabling:
  VLM detects "phase 1 complete" -> retrieve phase 2 keyframe -> inject as subgoal

Usage:
  # From eval videos (works for ALL tasks, no HDF5 needed)
  python build_library.py --mode video \
    --data_dir /path/to/eval_result/ar/ddp_causal \
    --output library.json

  # From eval videos, specific tasks
  python build_library.py --mode video \
    --data_dir /path/to/eval_result/ar/ddp_causal \
    --tasks adjust_bottle stack_bowls_two

  # From HDF5 (uses gripper actions for more accurate boundaries)
  python build_library.py --mode hdf5 \
    --data_dir /path/to/hdf5 \
    --tasks stack_bowls_two

  # Add VLM phase labels
  python build_library.py --mode video --data_dir /path/to/videos \
    --label_phases --output library_labeled.json
"""

import os
import sys
import json
import argparse
import glob
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "gt_keyframe_test"))
from extract_keyframes import KeyframeInfo, frame_to_base64

DEFAULT_VIDEO_DIR = "/mnt/shared-storage-user/kangli/workspace/cyujie/code/vidar-robotwin/eval_result/ar/ddp_causal"
DEFAULT_HDF5_DIR = "/mnt/shared-storage-user/kangli/workspace/cyujie/cyujie/datasets/RoboTwin2.0/dataset/processed/hdf5"
DEFAULT_MODEL_PATH = "/mnt/shared-storage-user/kangli/workspace/cyujie/mounts/ailab-public-shared/models--Qwen--Qwen2.5-VL-7B-Instruct"
GRIPPER_INDICES = (7, 15)  # binary gripper command columns (0=open, 1=close)
FPS = 10


# ─── Phase Boundary Detection (Video-based) ──────────────────────

def detect_boundaries_visual(
    frames: List[np.ndarray],
    motion_threshold: float = 0.008,
    change_threshold: float = 0.015,
    min_interval: int = 5,
) -> List[int]:
    """
    Detect phase boundaries from video frames using motion stops + visual changes.
    No action data needed — works on any video.

    Pass 1: Compute frame-to-frame motion (grayscale MAD).
    Pass 2: Detect motion stops (moving -> stationary transitions)
             and large visual changes.
    """
    T = len(frames)
    if T < 2:
        return [0, T - 1] if T > 0 else [0]

    motion_scores = np.zeros(T)
    change_scores = np.zeros(T)
    prev_gray = None

    for idx in range(T):
        gray = cv2.cvtColor(frames[idx], cv2.COLOR_BGR2GRAY).astype(float) / 255.0
        if prev_gray is not None:
            motion_scores[idx] = np.mean(np.abs(gray - prev_gray))
            change_scores[idx] = np.mean((gray - prev_gray) ** 2)
        prev_gray = gray

    # Smooth motion
    kernel = 3
    if T > kernel:
        smoothed = np.convolve(motion_scores, np.ones(kernel) / kernel, mode='same')
    else:
        smoothed = motion_scores

    # Print diagnostics
    ms = motion_scores[1:]
    if len(ms) > 0:
        print(f"    motion MAD: mean={ms.mean():.5f}, p50={np.percentile(ms, 50):.5f}, "
              f"p90={np.percentile(ms, 90):.5f}")

    # Detect boundaries
    boundaries = [0]
    last_b = 0
    was_moving = False

    for idx in range(1, T):
        if idx - last_b < min_interval:
            was_moving = smoothed[idx] > motion_threshold
            continue

        is_moving = smoothed[idx] > motion_threshold

        # Motion stop (moving -> stationary)
        if was_moving and not is_moving:
            boundaries.append(idx)
            last_b = idx
        # Large visual change
        elif change_scores[idx] > change_threshold:
            boundaries.append(idx)
            last_b = idx

        was_moving = is_moving

    if boundaries[-1] != T - 1:
        boundaries.append(T - 1)

    return boundaries


def extract_video_frames(video_path: str) -> List[np.ndarray]:
    """Load all frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def frame_at_index(frames: List[np.ndarray], idx: int) -> str:
    """Get base64-encoded frame at index."""
    img = frames[min(idx, len(frames) - 1)]
    # BGR -> RGB for consistency
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return frame_to_base64(img_rgb)


# ─── Phase Boundary Detection (HDF5-based) ───────────────────────

def detect_boundaries_gripper(
    actions: np.ndarray,
    gripper_indices: Tuple[int, ...] = GRIPPER_INDICES,
    threshold: float = 0.5,
    min_interval: int = 4,
) -> List[int]:
    """
    Detect phase boundaries from gripper state crossings.

    For binary gripper columns (0=open, 1=close), detects when the
    gripper state crosses the threshold (default 0.5), i.e., open→close
    or close→open transitions.

    For continuous gripper angles, set a lower threshold and uses
    per-frame delta detection as fallback.
    """
    T = len(actions)
    boundaries = [0]
    last_b = 0

    # Determine if gripper columns are binary-like
    is_binary = {}
    for g_idx in gripper_indices:
        if g_idx < actions.shape[1]:
            vals = actions[:, g_idx]
            unique_approx = len(np.unique(np.round(vals, 1)))
            is_binary[g_idx] = (unique_approx <= 5 and vals.max() <= 1.1 and vals.min() >= -0.1)

    for t in range(1, T):
        if t - last_b < min_interval:
            continue
        triggered = False
        for g_idx in gripper_indices:
            if g_idx >= actions.shape[1]:
                continue
            if is_binary.get(g_idx, False):
                # State crossing detection: open(< threshold) <-> close(>= threshold)
                prev_closed = actions[t - 1, g_idx] >= threshold
                curr_closed = actions[t, g_idx] >= threshold
                if prev_closed != curr_closed:
                    triggered = True
                    break
            else:
                # Fallback: cumulative change from last boundary
                change = abs(actions[t, g_idx] - actions[last_b, g_idx])
                if change > 0.15:
                    triggered = True
                    break

        if triggered:
            boundaries.append(t)
            last_b = t

    if boundaries[-1] != T - 1:
        boundaries.append(T - 1)
    return boundaries


# ─── Task Library Builder (Video mode) ───────────────────────────

def build_task_from_videos(
    task_dir: Path,
    task_name: str,
    max_episodes: int = 50,
) -> Optional[Dict]:
    """Build keyframe library for one task from eval result videos."""
    videos = sorted(task_dir.glob("episode*.mp4"))
    if not videos:
        print(f"  No videos found in {task_dir}")
        return None

    videos = videos[:max_episodes]
    print(f"  Found {len(videos)} videos")

    episode_data = {}
    all_boundaries = []

    for vp in videos:
        ep_name = vp.stem
        try:
            frames = extract_video_frames(str(vp))
            if len(frames) < 3:
                print(f"    {ep_name}: too short ({len(frames)} frames), skipping")
                continue

            boundaries = detect_boundaries_visual(frames)
            keyframe_b64s = [frame_at_index(frames, b) for b in boundaries]

            episode_data[ep_name] = {
                "boundaries": boundaries,
                "num_frames": len(frames),
                "num_phases": len(boundaries) - 1,
                "keyframe_b64s": keyframe_b64s,
            }
            all_boundaries.append(boundaries)
            print(f"    {ep_name}: {len(frames)} frames, "
                  f"{len(boundaries) - 1} phases, boundaries={boundaries}")

        except Exception as e:
            print(f"    {ep_name}: error - {e}")
            continue

    if not episode_data:
        return None

    return _aggregate_episodes(task_name, episode_data, all_boundaries)


# ─── Task Library Builder (HDF5 mode) ────────────────────────────

def build_task_from_hdf5(
    task_dir: Path,
    task_name: str,
    max_episodes: int = 50,
    gripper_threshold: float = 0.3,
) -> Optional[Dict]:
    """Build keyframe library for one task from HDF5 files."""
    import h5py

    episodes = sorted(task_dir.glob("episode_*.hdf5"))
    if not episodes:
        # Try without underscore
        episodes = sorted(task_dir.glob("episode*.hdf5"))
    if not episodes:
        print(f"  No HDF5 files found in {task_dir}")
        return None

    episodes = episodes[:max_episodes]
    print(f"  Found {len(episodes)} HDF5 files")

    episode_data = {}
    all_boundaries = []

    for ep_path in episodes:
        ep_name = ep_path.stem
        try:
            with h5py.File(ep_path, "r") as f:
                # Load actions
                if "action" in f:
                    actions = f["action"][:]
                elif "actions" in f:
                    actions = f["actions"][:]
                else:
                    print(f"    {ep_name}: no action data, skipping")
                    continue

                boundaries = detect_boundaries_gripper(
                    actions, threshold=gripper_threshold
                )
                total_frames = len(actions)

                # Load images at boundary frames
                if "observations/unified_image" in f:
                    images = f["observations/unified_image"]
                elif "observations/images/cam_high" in f:
                    images = f["observations/images/cam_high"]
                else:
                    print(f"    {ep_name}: no image data, skipping")
                    continue

                keyframe_b64s = []
                for b_idx in boundaries:
                    b_idx = min(b_idx, len(images) - 1)
                    img = images[b_idx]
                    if img.ndim == 2:
                        img = np.stack([img] * 3, axis=-1)
                    elif img.shape[0] == 3:
                        img = np.transpose(img, (1, 2, 0))
                    keyframe_b64s.append(frame_to_base64(img))

                instruction = ""
                if "instruction" in f.attrs:
                    instruction = str(f.attrs["instruction"])

            episode_data[ep_name] = {
                "boundaries": boundaries,
                "num_frames": total_frames,
                "num_phases": len(boundaries) - 1,
                "keyframe_b64s": keyframe_b64s,
                "instruction": instruction,
            }
            all_boundaries.append(boundaries)
            print(f"    {ep_name}: {total_frames} frames, "
                  f"{len(boundaries) - 1} phases, boundaries={boundaries}")

        except Exception as e:
            print(f"    {ep_name}: error - {e}")
            continue

    if not episode_data:
        return None

    return _aggregate_episodes(task_name, episode_data, all_boundaries)


# ─── Shared Aggregation ──────────────────────────────────────────

def _aggregate_episodes(
    task_name: str,
    episode_data: Dict,
    all_boundaries: List[List[int]],
) -> Dict:
    """Aggregate per-episode boundaries into canonical phases."""
    phase_counts = [len(b) - 1 for b in all_boundaries]
    most_common_num = Counter(phase_counts).most_common(1)[0][0]
    print(f"  Canonical phase count: {most_common_num} "
          f"(distribution: {dict(Counter(phase_counts))})")

    # Keep only episodes with canonical phase count
    canonical_episodes = {
        name: data for name, data in episode_data.items()
        if data["num_phases"] == most_common_num
    }

    if not canonical_episodes:
        # Fallback: use all episodes
        canonical_episodes = episode_data

    # Median boundary positions
    boundary_matrix = np.array([
        d["boundaries"][:most_common_num + 1]
        for d in canonical_episodes.values()
        if len(d["boundaries"]) >= most_common_num + 1
    ])
    if len(boundary_matrix) == 0:
        return None
    median_boundaries = np.median(boundary_matrix, axis=0).astype(int).tolist()

    # Build phase entries
    phases = []
    for phase_id in range(most_common_num):
        phase_keyframes = []
        for ep_name, ep_data in canonical_episodes.items():
            kf_idx = phase_id + 1  # boundary at end of this phase
            if kf_idx < len(ep_data["keyframe_b64s"]):
                phase_keyframes.append({
                    "episode": ep_name,
                    "frame_index": ep_data["boundaries"][kf_idx],
                    "image_b64": ep_data["keyframe_b64s"][kf_idx],
                })

        phases.append({
            "phase_id": phase_id,
            "start_frame": median_boundaries[phase_id],
            "end_frame": median_boundaries[phase_id + 1] if phase_id + 1 < len(median_boundaries) else median_boundaries[-1],
            "num_keyframes": len(phase_keyframes),
            "keyframes": phase_keyframes,
        })

    # Start keyframes (initial observation)
    start_keyframes = []
    for ep_name, ep_data in canonical_episodes.items():
        if ep_data["keyframe_b64s"]:
            start_keyframes.append({
                "episode": ep_name,
                "frame_index": 0,
                "image_b64": ep_data["keyframe_b64s"][0],
            })

    return {
        "task": task_name,
        "num_phases": most_common_num,
        "median_boundaries": median_boundaries,
        "total_episodes": len(episode_data),
        "canonical_episodes": len(canonical_episodes),
        "start_keyframes": start_keyframes,
        "phases": phases,
        "episodes": {
            name: {k: v for k, v in d.items() if k != "keyframe_b64s"}
            for name, d in episode_data.items()
        },
    }


# ─── Full Library Builder ────────────────────────────────────────

def build_full_library(
    data_dir: str,
    mode: str = "video",
    tasks: Optional[List[str]] = None,
    max_episodes: int = 50,
    gripper_threshold: float = 0.3,
) -> Dict:
    """Build keyframe library for all tasks."""
    data_path = Path(data_dir)

    if tasks:
        task_dirs = [(data_path / t, t) for t in tasks if (data_path / t).exists()]
    else:
        task_dirs = [
            (d, d.name) for d in sorted(data_path.iterdir())
            if d.is_dir() and not d.name.startswith(".")
        ]

    print(f"Building library ({mode} mode) for {len(task_dirs)} tasks from {data_dir}\n")
    library = {}

    for task_dir, task_name in task_dirs:
        print(f"[{task_name}]")
        if mode == "hdf5":
            result = build_task_from_hdf5(
                task_dir, task_name,
                max_episodes=max_episodes,
                gripper_threshold=gripper_threshold,
            )
        else:  # video
            result = build_task_from_videos(
                task_dir, task_name,
                max_episodes=max_episodes,
            )

        if result:
            library[task_name] = result
            print(f"  => {result['num_phases']} phases, "
                  f"{result['canonical_episodes']}/{result['total_episodes']} canonical\n")
        else:
            print(f"  => FAILED\n")

    return library


# ─── VLM Phase Labeling (optional) ───────────────────────────────

def label_phases_with_vlm(library: Dict, model_path: str) -> Dict:
    """Use Qwen2.5-VL-7B to generate semantic phase names."""
    from PIL import Image
    import io
    import base64
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    sys.path.insert(0, "/mnt/shared-storage-user/kangli/workspace/cyujie/code/vidar/wan/utils")
    from qwen_vl_utils import process_vision_info

    print(f"\nLoading VLM for phase labeling...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="flash_attention_2",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    for task_name, task_data in library.items():
        print(f"\nLabeling phases for [{task_name}]...")

        phase_images = []
        for phase in task_data["phases"]:
            if phase["keyframes"]:
                img_b64 = phase["keyframes"][0]["image_b64"]
                img_bytes = base64.b64decode(img_b64)
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                phase_images.append(pil_img)

        if not phase_images:
            continue

        if task_data["start_keyframes"]:
            start_b64 = task_data["start_keyframes"][0]["image_b64"]
            start_img = Image.open(io.BytesIO(base64.b64decode(start_b64))).convert("RGB")
        else:
            start_img = phase_images[0]

        content = [
            {"type": "text", "text": f"A dual-arm robot is performing task: \"{task_name}\"\n"
             f"Below are {len(phase_images) + 1} images showing the robot at each phase boundary "
             f"(initial state, then end of each phase).\n\n"
             f"Image 0 (initial state):"},
            {"type": "image", "image": start_img},
        ]
        for i, img in enumerate(phase_images):
            content.append({"type": "text", "text": f"\nImage {i+1} (end of phase {i}):"})
            content.append({"type": "image", "image": img})

        content.append({"type": "text", "text": """
Give each phase a short descriptive name (e.g., "approach_bowl", "grasp", "place").
Output as JSON:
{
  "phases": [
    {"phase_id": 0, "name": "phase_name", "description": "what happens"}
  ]
}"""})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": "You are a robot task analysis expert. Output valid JSON only."}]},
            {"role": "user", "content": content},
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=512)
        out_ids = out_ids[:, inputs.input_ids.shape[1]:]
        response = processor.batch_decode(out_ids, skip_special_tokens=True)[0]

        try:
            txt = response.strip()
            if txt.startswith("```"):
                txt = txt.split("\n", 1)[1]
                txt = txt.rsplit("```", 1)[0]
            parsed = json.loads(txt)
            phase_names = [p.get("name", f"phase_{p['phase_id']}") for p in parsed["phases"]]
            task_data["phase_names"] = phase_names
            print(f"  Phases: {phase_names}")

            for i, p in enumerate(parsed["phases"]):
                if i < len(task_data["phases"]):
                    task_data["phases"][i]["name"] = p.get("name", "")
                    task_data["phases"][i]["description"] = p.get("description", "")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Failed to parse VLM response: {e}")
            print(f"  Raw: {response[:200]}")

    return library


# ─── Library I/O ──────────────────────────────────────────────────

def save_library(library: Dict, output_path: str, compact: bool = False):
    """Save library to JSON. Use compact=True to exclude keyframe images."""
    if compact:
        lib_copy = {}
        for task, data in library.items():
            task_copy = {k: v for k, v in data.items() if k not in ("phases", "start_keyframes")}
            task_copy["start_keyframes"] = [
                {k: v for k, v in kf.items() if k != "image_b64"}
                for kf in data.get("start_keyframes", [])
            ]
            task_copy["phases"] = []
            for phase in data.get("phases", []):
                p_copy = {k: v for k, v in phase.items() if k != "keyframes"}
                p_copy["keyframes"] = [
                    {k: v for k, v in kf.items() if k != "image_b64"}
                    for kf in phase.get("keyframes", [])
                ]
                task_copy["phases"].append(p_copy)
            lib_copy[task] = task_copy
        library = lib_copy

    with open(output_path, "w") as f:
        json.dump(library, f, indent=2, ensure_ascii=False)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nLibrary saved to {output_path} ({size_mb:.1f} MB)")


def load_library(library_path: str) -> Dict:
    """Load library from JSON."""
    with open(library_path) as f:
        return json.load(f)


# ─── Runtime Retrieval ────────────────────────────────────────────

class KeyframeLibrary:
    """
    Runtime interface for Method B keyframe retrieval.

    Usage:
        lib = KeyframeLibrary("library.json")
        kf = lib.get_phase_keyframe("stack_bowls_two", phase_id=1)
    """

    def __init__(self, library_path: str):
        self.library = load_library(library_path)

    def get_tasks(self) -> List[str]:
        return list(self.library.keys())

    def get_num_phases(self, task: str) -> int:
        return self.library[task]["num_phases"]

    def get_phase_keyframe(self, task: str, phase_id: int,
                           episode_idx: int = 0) -> Optional[str]:
        """Get keyframe image (base64) for a specific phase."""
        phases = self.library[task]["phases"]
        if phase_id >= len(phases):
            return None
        kfs = phases[phase_id]["keyframes"]
        if not kfs or episode_idx >= len(kfs):
            return None
        return kfs[episode_idx].get("image_b64")

    def get_all_phase_keyframes(self, task: str,
                                episode_idx: int = 0) -> List[str]:
        """Get keyframes for all phases (for full subgoal sequence)."""
        phases = self.library[task]["phases"]
        result = []
        for phase in phases:
            kfs = phase["keyframes"]
            if kfs and episode_idx < len(kfs):
                b64 = kfs[episode_idx].get("image_b64")
                if b64:
                    result.append(b64)
        return result

    def get_phase_info(self, task: str) -> List[Dict]:
        """Get metadata for all phases (no images)."""
        return [
            {k: v for k, v in p.items() if k != "keyframes"}
            for p in self.library[task]["phases"]
        ]


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build GT Keyframe Library for Method B")
    parser.add_argument("--mode", type=str, default="video", choices=["video", "hdf5"],
                        help="Data source: 'video' (mp4, all tasks) or 'hdf5' (gripper-based)")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Data directory (default depends on mode)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Specific tasks to process (default: all)")
    parser.add_argument("--output", type=str, default="library.json",
                        help="Output library JSON path")
    parser.add_argument("--compact", action="store_true",
                        help="Save compact version without images (metadata only)")
    parser.add_argument("--max_episodes", type=int, default=50,
                        help="Max episodes per task")
    parser.add_argument("--gripper_threshold", type=float, default=0.3,
                        help="Gripper state change threshold (hdf5 mode)")
    parser.add_argument("--label_phases", action="store_true",
                        help="Use VLM to generate semantic phase names")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Qwen2.5-VL model path (for --label_phases)")
    args = parser.parse_args()

    # Default data_dir depends on mode
    if args.data_dir is None:
        args.data_dir = DEFAULT_VIDEO_DIR if args.mode == "video" else DEFAULT_HDF5_DIR

    t0 = time.time()
    library = build_full_library(
        args.data_dir,
        mode=args.mode,
        tasks=args.tasks,
        max_episodes=args.max_episodes,
        gripper_threshold=args.gripper_threshold,
    )

    if not library:
        print("No tasks processed. Check --data_dir path.")
        sys.exit(1)

    if args.label_phases:
        library = label_phases_with_vlm(library, args.model_path)

    save_library(library, args.output, compact=args.compact)

    # Summary
    print(f"\n{'='*60}")
    print(f"Library Summary ({time.time() - t0:.1f}s)")
    print(f"{'='*60}")
    for task_name, data in library.items():
        names = data.get("phase_names", [])
        name_str = f" ({', '.join(names)})" if names else ""
        print(f"  {task_name}: {data['num_phases']} phases{name_str}, "
              f"{data['canonical_episodes']}/{data['total_episodes']} canonical")


if __name__ == "__main__":
    main()
