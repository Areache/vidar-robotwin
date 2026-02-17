#!/usr/bin/env python
"""
TI2V Model Keyframe Generation Quality — Quick Test & Full Experiment.

Generates subgoal images using Wan2.2-TI2V-5B and compares with GT keyframes.
See subgoal_keyframe_model.md for the full experiment design.

Quick test (1 episode, 3 subgoals, cfg=5.0, seed=42):
    python experiments/subgoal/run_keyframe_model.py \
        --hdf5 /path/to/episode.hdf5 \
        --mode quick

Full experiment (5 episodes, 3 subgoals, 3 cfgs, 3 seeds = 135 gens):
    python experiments/subgoal/run_keyframe_model.py \
        --data-dir /path/to/hdf5/ \
        --mode full

Dry run (verify data paths without loading model):
    python experiments/subgoal/run_keyframe_model.py \
        --hdf5 /path/to/episode.hdf5 \
        --dry-run
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict, field

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

# Add key_frame_wan to path for WanSubgoalGenerator
KEY_FRAME_WAN_DIR = PROJECT_ROOT / "experiments" / "key_frame_wan" / "stack_bowl_two"
sys.path.insert(0, str(KEY_FRAME_WAN_DIR))

import h5py

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT = "/mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/vidar/Wan2.2-TI2V-5B"
DEFAULT_DATA_DIR = "/mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/datasets/robotwin/processed/hdf5"

# Subgoals for stack_bowls_two (from subgoals.json)
TASK_SUBGOALS = {
    "stack_bowls_two": [
        {
            "subgoal_id": 1,
            "subgoal_name": "bowl_A_grasped",
            "semantic_image_description": (
                "The robot gripper is holding bowl {A}, lifted above the table surface. "
                "Bowl {B} remains on the table in its original position."
            ),
            "key_conditions": [
                "Bowl {A} is gripped by the robot",
                "Bowl {A} is elevated from the table",
                "Bowl {B} is stationary on the table",
            ],
        },
        {
            "subgoal_id": 2,
            "subgoal_name": "bowl_A_positioned_above_B",
            "semantic_image_description": (
                "Bowl {A} is held by the robot gripper, positioned directly above bowl {B}. "
                "Both bowls are aligned vertically."
            ),
            "key_conditions": [
                "Bowl {A} is above bowl {B}",
                "Bowls are vertically aligned",
                "Bowl {B} is on the table",
            ],
        },
        {
            "subgoal_id": 3,
            "subgoal_name": "bowl_A_stacked_on_B",
            "semantic_image_description": (
                "Bowl {A} is resting on top of bowl {B}, forming a stable stack. "
                "The robot gripper has released bowl {A}."
            ),
            "key_conditions": [
                "Bowl {A} is on top of bowl {B}",
                "Stack is stable",
                "Robot gripper is open/released",
            ],
        },
    ],
}

# Experiment grid
GUIDE_SCALES = [3.0, 5.0, 7.0]
SEEDS = [42, 123, 456]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class GenerationResult:
    episode_id: str
    task: str
    subgoal_id: int
    subgoal_name: str
    prompt: str
    guide_scale: float
    seed: int
    generation_time_s: float = 0.0
    output_image_path: str = ""
    output_video_dir: str = ""
    metrics: Dict = field(default_factory=dict)


def find_episodes(data_dir: str, task_filter: Optional[str] = None, max_n: int = 5) -> List[Path]:
    """Find HDF5 episode files. Searches task subdirs if they exist."""
    data_path = Path(data_dir)
    files = []

    if task_filter:
        # Try task-specific subdirectory first
        task_dir = data_path / task_filter
        if task_dir.is_dir():
            files = sorted(task_dir.glob("episode_*.hdf5"))
            if not files:
                files = sorted(task_dir.glob("*.hdf5"))

    if not files:
        # Flat directory
        files = sorted(data_path.glob("episode_*.hdf5"))
        if not files:
            files = sorted(data_path.glob("*.hdf5"))

    # Deduplicate
    seen = set()
    unique = []
    for f in files:
        if f.name not in seen:
            seen.add(f.name)
            unique.append(f)
    return unique[:max_n]


def load_first_frame(hdf5_path: str) -> np.ndarray:
    """Load first frame from HDF5. Returns RGB (H, W, 3)."""
    with h5py.File(hdf5_path, "r") as f:
        if "observations/unified_image" in f:
            return f["observations/unified_image"][0]
        elif "observations/images/cam_high" in f:
            return f["observations/images/cam_high"][0]
    raise ValueError(f"No image data found in {hdf5_path}")


def load_hdf5_info(hdf5_path: str) -> Dict:
    """Load episode metadata (frame count, instruction, etc.)."""
    with h5py.File(hdf5_path, "r") as f:
        info = {"path": str(hdf5_path)}

        # Frame count
        if "observations/unified_image" in f:
            info["n_frames"] = f["observations/unified_image"].shape[0]
            info["image_shape"] = f["observations/unified_image"].shape[1:]
        elif "observations/images/cam_high" in f:
            info["n_frames"] = f["observations/images/cam_high"].shape[0]
            info["image_shape"] = f["observations/images/cam_high"].shape[1:]

        # Instruction
        if "instruction" in f.attrs:
            instr = f.attrs["instruction"]
            info["instruction"] = instr.decode("utf-8") if isinstance(instr, bytes) else str(instr)
        else:
            info["instruction"] = ""

        # Action shape
        if "action" in f:
            info["action_shape"] = f["action"].shape
    return info


# ---------------------------------------------------------------------------
# Generator wrapper
# ---------------------------------------------------------------------------
class SubgoalGenerator:
    """Wrapper around WanSubgoalGenerator with structured output."""

    def __init__(self, checkpoint_dir: str, device_id: int = 0):
        self.checkpoint_dir = checkpoint_dir
        self.device_id = device_id
        self.model = None

    def load_model(self):
        if self.model is not None:
            return

        from wan.configs.wan_ti2v_5B import wan_ti2v_5B
        from wan.textimage2video import WanTI2V

        print(f"Loading Wan2.2-TI2V-5B from {self.checkpoint_dir}...")
        self.model = WanTI2V(
            config=wan_ti2v_5B,
            checkpoint_dir=self.checkpoint_dir,
            device_id=self.device_id,
            t5_cpu=True,
            init_on_cpu=True,
        )
        print("Model loaded.")

    def generate(
        self,
        first_frame: np.ndarray,
        subgoal_description: str,
        frame_num: int = 17,
        sampling_steps: int = 30,
        guide_scale: float = 5.0,
        seed: int = 42,
    ) -> np.ndarray:
        """Generate video. Returns (T, H, W, 3) uint8 RGB."""
        self.load_model()

        pil_image = Image.fromarray(first_frame)

        prompt = (
            "The whole scene is in a realistic, industrial art style with three views: "
            "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
            f"The aloha robot achieves the following state: {subgoal_description}"
        )

        print(f"  Generating (cfg={guide_scale}, seed={seed}): {subgoal_description[:60]}...")

        video_tensor = self.model.generate(
            input_prompt=prompt,
            img=pil_image,
            frame_num=frame_num,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=True,
        )

        if video_tensor is None:
            raise RuntimeError("Video generation failed — model returned None")

        # (C, T, H, W) -> (T, H, W, C)
        video = video_tensor.permute(1, 2, 3, 0).cpu().numpy()
        video = ((video + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
        return video


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------
def save_video_frames(video: np.ndarray, out_dir: Path):
    """Save all frames of a generated video as individual images."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in range(len(video)):
        path = out_dir / f"frame_{t:03d}.jpg"
        cv2.imwrite(str(path), cv2.cvtColor(video[t], cv2.COLOR_RGB2BGR))


def vis1_gen_vs_first(
    first_frame: np.ndarray,
    subgoal_images: List[np.ndarray],
    subgoals: List[Dict],
    episode_name: str,
    out_path: Path,
):
    """
    Vis 1: First frame + generated subgoal images side-by-side.
    (GT keyframes omitted until rule-based extraction is done.)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_sg = len(subgoal_images)
    fig, axes = plt.subplots(n_sg, 2, figsize=(10, 4 * n_sg))
    if n_sg == 1:
        axes = [axes]

    fig.suptitle(f"Vis 1: Generated Subgoals — {episode_name}", fontsize=14, y=1.01)

    for i, (sg_img, sg_def) in enumerate(zip(subgoal_images, subgoals)):
        # First frame
        ax = axes[i][0]
        ax.imshow(first_frame)
        ax.set_title("First Frame (input)", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("Input", fontsize=10)

        # Generated subgoal
        ax = axes[i][1]
        ax.imshow(sg_img)
        ax.set_title(f"SG{sg_def['subgoal_id']}: {sg_def['subgoal_name']}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # Add description below
        desc = sg_def["semantic_image_description"]
        if len(desc) > 80:
            desc = desc[:77] + "..."
        ax.set_xlabel(desc, fontsize=7, wrap=True)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def vis2_video_strip(
    video: np.ndarray,
    subgoal_def: Dict,
    guide_scale: float,
    seed: int,
    out_path: Path,
):
    """Vis 2: Full 17-frame video strip for one generation."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(video)
    n_cols = min(T, 9)
    n_rows = (T + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.5 * n_rows))
    if n_rows == 1:
        axes = [axes]

    sg_name = subgoal_def["subgoal_name"]
    fig.suptitle(
        f"Vis 2: Video Strip — SG{subgoal_def['subgoal_id']}: {sg_name} "
        f"(cfg={guide_scale}, seed={seed})",
        fontsize=11,
        y=1.01,
    )

    for t in range(T):
        row, col = divmod(t, n_cols)
        ax = axes[row][col] if isinstance(axes[row], (list, np.ndarray)) else axes[row]
        ax.imshow(video[t])
        ax.set_title(f"f{t}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide empty cells
    for t in range(T, n_rows * n_cols):
        row, col = divmod(t, n_cols)
        ax = axes[row][col] if isinstance(axes[row], (list, np.ndarray)) else axes[row]
        ax.set_visible(False)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------
def run_quick_test(
    hdf5_path: str,
    checkpoint_dir: str,
    out_dir: Path,
    device_id: int = 0,
    guide_scale: float = 5.0,
    seed: int = 42,
    task: str = "stack_bowls_two",
):
    """
    Quick test: 1 episode × 3 subgoals × 1 cfg × 1 seed = 3 generations.
    Produces Vis 1 (side-by-side) and Vis 2 (video strips).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_name = Path(hdf5_path).stem

    print(f"{'='*60}")
    print(f"QUICK TEST: {episode_name}")
    print(f"  Task: {task}")
    print(f"  Guide scale: {guide_scale}")
    print(f"  Seed: {seed}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # Load episode info
    info = load_hdf5_info(hdf5_path)
    print(f"  Frames: {info.get('n_frames', '?')}")
    print(f"  Instruction: {info.get('instruction', '(none)')}")

    # Load first frame
    first_frame = load_first_frame(hdf5_path)
    print(f"  First frame shape: {first_frame.shape}")

    # Save first frame
    ff_path = out_dir / "first_frame.jpg"
    cv2.imwrite(str(ff_path), cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    print(f"  Saved first frame: {ff_path}")

    # Get subgoals
    subgoals = TASK_SUBGOALS.get(task)
    if subgoals is None:
        print(f"ERROR: No subgoals defined for task '{task}'")
        print(f"  Available tasks: {list(TASK_SUBGOALS.keys())}")
        sys.exit(1)

    # Save subgoals config
    with open(out_dir / "subgoals.json", "w") as f:
        json.dump(subgoals, f, indent=2)

    # Initialize generator
    generator = SubgoalGenerator(checkpoint_dir, device_id=device_id)

    # Generate subgoal images
    results = []
    subgoal_images = []
    videos = []

    for sg in subgoals:
        sg_id = sg["subgoal_id"]
        sg_name = sg["subgoal_name"]
        sg_desc = sg["semantic_image_description"]

        print(f"\n--- Subgoal {sg_id}: {sg_name} ---")

        t_start = time.time()
        video = generator.generate(
            first_frame=first_frame,
            subgoal_description=sg_desc,
            frame_num=17,
            sampling_steps=30,
            guide_scale=guide_scale,
            seed=seed,
        )
        gen_time = time.time() - t_start
        print(f"  Generated in {gen_time:.1f}s — video shape: {video.shape}")

        # Extract last frame as subgoal image
        sg_image = video[-1]
        subgoal_images.append(sg_image)
        videos.append(video)

        # Save subgoal image
        img_name = f"{episode_name}_sg{sg_id}_cfg{guide_scale}_seed{seed}.jpg"
        img_path = out_dir / "generated_images" / img_name
        img_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(img_path), cv2.cvtColor(sg_image, cv2.COLOR_RGB2BGR))

        # Save all 17 video frames
        vid_dir = out_dir / "generated_images" / f"{episode_name}_sg{sg_id}_cfg{guide_scale}_seed{seed}_video"
        save_video_frames(video, vid_dir)

        # Record result
        result = GenerationResult(
            episode_id=episode_name,
            task=task,
            subgoal_id=sg_id,
            subgoal_name=sg_name,
            prompt=sg_desc,
            guide_scale=guide_scale,
            seed=seed,
            generation_time_s=gen_time,
            output_image_path=str(img_path),
            output_video_dir=str(vid_dir),
        )
        results.append(result)
        print(f"  Saved: {img_path}")

    # --- Visualizations ---
    print(f"\n{'='*60}")
    print("Generating visualizations...")

    # Vis 1: First frame + generated subgoals
    vis1_gen_vs_first(
        first_frame, subgoal_images, subgoals, episode_name,
        out_dir / f"vis1_gen_vs_first_{episode_name}.png",
    )

    # Vis 2: Video strips (one per subgoal)
    for sg, video in zip(subgoals, videos):
        vis2_video_strip(
            video, sg, guide_scale, seed,
            out_dir / f"vis2_video_strip_{episode_name}_sg{sg['subgoal_id']}.png",
        )

    # Save generation log
    log = {
        "mode": "quick_test",
        "episode": episode_name,
        "task": task,
        "hdf5_path": str(hdf5_path),
        "checkpoint_dir": checkpoint_dir,
        "guide_scale": guide_scale,
        "seed": seed,
        "n_generations": len(results),
        "results": [asdict(r) for r in results],
    }
    log_path = out_dir / "generation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nGeneration log: {log_path}")

    # Summary
    total_time = sum(r.generation_time_s for r in results)
    print(f"\n{'='*60}")
    print(f"QUICK TEST COMPLETE")
    print(f"  Generations: {len(results)}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Avg per generation: {total_time / len(results):.1f}s")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    return results


def run_full_experiment(
    episode_paths: List[Path],
    checkpoint_dir: str,
    out_dir: Path,
    device_id: int = 0,
    task: str = "stack_bowls_two",
    guide_scales: List[float] = None,
    seeds: List[int] = None,
):
    """
    Full experiment: N episodes × 3 subgoals × 3 cfgs × 3 seeds.
    Produces Vis 1-3 (Vis 4-5 are post-hoc).
    """
    if guide_scales is None:
        guide_scales = GUIDE_SCALES
    if seeds is None:
        seeds = SEEDS

    out_dir.mkdir(parents=True, exist_ok=True)
    subgoals = TASK_SUBGOALS.get(task)
    if subgoals is None:
        print(f"ERROR: No subgoals defined for task '{task}'")
        sys.exit(1)

    n_total = len(episode_paths) * len(subgoals) * len(guide_scales) * len(seeds)
    print(f"{'='*60}")
    print(f"FULL EXPERIMENT")
    print(f"  Episodes: {len(episode_paths)}")
    print(f"  Subgoals: {len(subgoals)}")
    print(f"  Guide scales: {guide_scales}")
    print(f"  Seeds: {seeds}")
    print(f"  Total generations: {n_total}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    generator = SubgoalGenerator(checkpoint_dir, device_id=device_id)
    all_results = []
    gen_count = 0

    for ep_path in episode_paths:
        episode_name = ep_path.stem
        first_frame = load_first_frame(str(ep_path))
        print(f"\n{'='*40}")
        print(f"Episode: {episode_name} — frame shape: {first_frame.shape}")

        # Save first frame
        ep_out = out_dir / "generated_images"
        ep_out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(ep_out / f"{episode_name}_first_frame.jpg"),
            cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR),
        )

        # For Vis 1: collect best subgoal images (cfg=5.0, seed=42)
        default_sg_images = []

        for sg in subgoals:
            sg_id = sg["subgoal_id"]
            sg_name = sg["subgoal_name"]
            sg_desc = sg["semantic_image_description"]

            # For Vis 3: collect images across cfg/seed
            cfg_seed_images = {}

            for cfg in guide_scales:
                for seed in seeds:
                    gen_count += 1
                    print(f"\n[{gen_count}/{n_total}] {episode_name} sg{sg_id} cfg={cfg} seed={seed}")

                    t_start = time.time()
                    video = generator.generate(
                        first_frame=first_frame,
                        subgoal_description=sg_desc,
                        frame_num=17,
                        sampling_steps=30,
                        guide_scale=cfg,
                        seed=seed,
                    )
                    gen_time = time.time() - t_start

                    sg_image = video[-1]

                    # Save image
                    img_name = f"{episode_name}_sg{sg_id}_cfg{cfg}_seed{seed}.jpg"
                    img_path = ep_out / img_name
                    cv2.imwrite(str(img_path), cv2.cvtColor(sg_image, cv2.COLOR_RGB2BGR))

                    # Save video frames
                    vid_dir = ep_out / f"{episode_name}_sg{sg_id}_cfg{cfg}_seed{seed}_video"
                    save_video_frames(video, vid_dir)

                    result = GenerationResult(
                        episode_id=episode_name,
                        task=task,
                        subgoal_id=sg_id,
                        subgoal_name=sg_name,
                        prompt=sg_desc,
                        guide_scale=cfg,
                        seed=seed,
                        generation_time_s=gen_time,
                        output_image_path=str(img_path),
                        output_video_dir=str(vid_dir),
                    )
                    all_results.append(result)
                    cfg_seed_images[(cfg, seed)] = sg_image

                    # Collect default for Vis 1
                    if cfg == 5.0 and seed == 42:
                        default_sg_images.append(sg_image)

                    print(f"  Done in {gen_time:.1f}s — saved: {img_path}")

            # Vis 3: CFG comparison grid for this subgoal
            vis3_cfg_comparison(
                cfg_seed_images, sg, guide_scales, seeds,
                episode_name,
                out_dir / f"vis3_cfg_{episode_name}_sg{sg_id}.png",
            )

            # Vis 2: Video strip for default config
            default_video_dir = ep_out / f"{episode_name}_sg{sg_id}_cfg5.0_seed42_video"
            if default_video_dir.exists():
                frames = []
                for t in range(17):
                    fp = default_video_dir / f"frame_{t:03d}.jpg"
                    if fp.exists():
                        img = cv2.cvtColor(cv2.imread(str(fp)), cv2.COLOR_BGR2RGB)
                        frames.append(img)
                if frames:
                    vis2_video_strip(
                        np.array(frames), sg, 5.0, 42,
                        out_dir / f"vis2_video_strip_{episode_name}_sg{sg_id}.png",
                    )

        # Vis 1: Side-by-side for this episode
        if default_sg_images:
            vis1_gen_vs_first(
                first_frame, default_sg_images, subgoals, episode_name,
                out_dir / f"vis1_gen_vs_first_{episode_name}.png",
            )

    # Save full generation log
    log = {
        "mode": "full",
        "task": task,
        "episodes": [str(p) for p in episode_paths],
        "guide_scales": guide_scales,
        "seeds": seeds,
        "n_generations": len(all_results),
        "results": [asdict(r) for r in all_results],
    }
    log_path = out_dir / "generation_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    total_time = sum(r.generation_time_s for r in all_results)
    print(f"\n{'='*60}")
    print(f"FULL EXPERIMENT COMPLETE")
    print(f"  Total generations: {len(all_results)}")
    print(f"  Total time: {total_time:.0f}s ({total_time/3600:.1f}h)")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    return all_results


def vis3_cfg_comparison(
    cfg_seed_images: Dict[Tuple[float, int], np.ndarray],
    subgoal_def: Dict,
    guide_scales: List[float],
    seeds: List[int],
    episode_name: str,
    out_path: Path,
):
    """Vis 3: Guide scale × seed grid for one subgoal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(seeds)
    n_cols = len(guide_scales)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[a] for a in axes]

    sg_name = subgoal_def["subgoal_name"]
    fig.suptitle(
        f"Vis 3: CFG Comparison — {episode_name} / SG{subgoal_def['subgoal_id']}: {sg_name}",
        fontsize=12,
        y=1.01,
    )

    for r, seed in enumerate(seeds):
        for c, cfg in enumerate(guide_scales):
            ax = axes[r][c]
            key = (cfg, seed)
            if key in cfg_seed_images:
                ax.imshow(cfg_seed_images[key])
            else:
                ax.text(0.5, 0.5, "N/A", transform=ax.transAxes, ha="center")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(f"cfg={cfg}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"seed={seed}", fontsize=10)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="TI2V Model Keyframe Generation Quality Experiment"
    )

    # Input
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--hdf5", type=str, help="Single HDF5 episode file")
    input_group.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="HDF5 data directory")

    # Model
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device-id", type=int, default=0)

    # Experiment
    parser.add_argument("--mode", choices=["quick", "full"], default="quick",
                        help="quick=1ep×3sg×1cfg×1seed, full=Nep×3sg×3cfg×3seed")
    parser.add_argument("--task", type=str, default="stack_bowls_two")
    parser.add_argument("--max-episodes", type=int, default=5)
    parser.add_argument("--guide-scale", type=float, default=5.0, help="CFG for quick mode")
    parser.add_argument("--seed", type=int, default=42, help="Seed for quick mode")

    # Output
    parser.add_argument("--out-dir", type=str, default=None)

    # Debug
    parser.add_argument("--dry-run", action="store_true", help="Check data paths only, don't load model")

    args = parser.parse_args()

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        mode_suffix = "quick_test" if args.mode == "quick" else "full"
        out_dir = SCRIPT_DIR / "results" / "model" / mode_suffix

    # Resolve episode paths
    if args.hdf5:
        episode_paths = [Path(args.hdf5)]
    else:
        episode_paths = find_episodes(args.data_dir, task_filter=args.task, max_n=args.max_episodes)

    if not episode_paths:
        print(f"ERROR: No HDF5 files found.")
        print(f"  Tried: {args.data_dir}")
        print(f"  Task filter: {args.task}")
        print(f"\nUsage:")
        print(f"  python {__file__} --hdf5 /path/to/episode.hdf5 --mode quick")
        print(f"  python {__file__} --data-dir /path/to/hdf5/ --mode full")
        sys.exit(1)

    # Verify files exist
    for p in episode_paths:
        if not p.exists():
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    # Dry run
    if args.dry_run:
        print(f"DRY RUN — checking data only (no model loading)")
        print(f"Mode: {args.mode}")
        print(f"Checkpoint: {args.checkpoint_dir}")
        print(f"  Exists: {Path(args.checkpoint_dir).exists()}")
        print(f"Task: {args.task}")
        print(f"Subgoals defined: {args.task in TASK_SUBGOALS}")
        print(f"\nEpisodes ({len(episode_paths)}):")
        for p in episode_paths:
            try:
                info = load_hdf5_info(str(p))
                print(f"  {p.name}: T={info.get('n_frames', '?')}, "
                      f"shape={info.get('image_shape', '?')}, "
                      f"instr='{info.get('instruction', '')[:60]}'")
            except Exception as e:
                print(f"  {p.name}: ERROR — {e}")

        if args.mode == "quick":
            n_gens = len(TASK_SUBGOALS.get(args.task, [])) * 1 * 1
        else:
            n_gens = (
                len(episode_paths)
                * len(TASK_SUBGOALS.get(args.task, []))
                * len(GUIDE_SCALES)
                * len(SEEDS)
            )
        print(f"\nTotal generations planned: {n_gens}")
        print(f"Output would be: {out_dir}")
        return

    # Run experiment
    if args.mode == "quick":
        run_quick_test(
            hdf5_path=str(episode_paths[0]),
            checkpoint_dir=args.checkpoint_dir,
            out_dir=out_dir,
            device_id=args.device_id,
            guide_scale=args.guide_scale,
            seed=args.seed,
            task=args.task,
        )
    else:
        run_full_experiment(
            episode_paths=episode_paths,
            checkpoint_dir=args.checkpoint_dir,
            out_dir=out_dir,
            device_id=args.device_id,
            task=args.task,
        )


if __name__ == "__main__":
    main()
