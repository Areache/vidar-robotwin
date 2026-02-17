#!/usr/bin/env python
"""
Minimal rule-based keyframe extraction experiment.

Extracts keyframes from HDF5 demonstrations using 3 strategies
(uniform, gripper_change, visual_change) and generates 5 diagnostic
visualizations to verify correctness.

Usage:
    # Auto-discover HDF5 files from default path
    python experiments/subgoal/run_keyframe_rule.py

    # Specify HDF5 directory
    python experiments/subgoal/run_keyframe_rule.py --data-dir /path/to/hdf5

    # Specify individual files
    python experiments/subgoal/run_keyframe_rule.py --episodes ep1.hdf5 ep2.hdf5

    # Dry run (just list episodes, don't extract)
    python experiments/subgoal/run_keyframe_rule.py --dry-run
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import asdict

# Add project root to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # vidar-robotwin
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "gt_keyframe_test"))

import h5py
import cv2

# Reuse existing extraction functions
from extract_keyframes import (
    KeyframeInfo,
    extract_keyframes_from_hdf5,
    base64_to_numpy,
    frame_to_base64,
    visualize_keyframes,
    visualize_keyframes_timeline,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/datasets/robotwin/processed/hdf5"

STRATEGIES = {
    "uniform":          dict(strategy="uniform", interval=8),
    "gripper":          dict(strategy="gripper", gripper_threshold=0.3),
    "visual":           dict(strategy="visual_change"),
    "action_milestone": dict(strategy="action_milestone"),
    "semantic":         dict(strategy="semantic"),
    "composite":        dict(strategy="composite"),
}

MAX_KEYFRAMES = 20
MAX_EPISODES = 5  # pick at most 5 episodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_episodes(data_dir: str, max_n: int = MAX_EPISODES):
    """Find HDF5 episode files, return up to max_n sorted by name."""
    data_path = Path(data_dir)
    patterns = ["episode_*.hdf5", "*.hdf5"]
    files = []
    for p in patterns:
        files.extend(sorted(data_path.glob(p)))
        if files:
            break
    # Deduplicate and take first max_n
    seen = set()
    unique = []
    for f in files:
        if f.name not in seen:
            seen.add(f.name)
            unique.append(f)
    return unique[:max_n]


def load_hdf5_data(hdf5_path: str):
    """Load images, actions, and instruction from HDF5."""
    with h5py.File(hdf5_path, "r") as f:
        # Images
        if "observations/unified_image" in f:
            images = f["observations/unified_image"][:]
        elif "observations/images/cam_high" in f:
            images = f["observations/images/cam_high"][:]
        else:
            raise ValueError(f"No image data in {hdf5_path}")

        # Actions
        actions = None
        if "action" in f:
            actions = f["action"][:]
        elif "actions" in f:
            actions = f["actions"][:]

        # Instruction
        instruction = ""
        if "instruction" in f.attrs:
            instruction = f.attrs["instruction"]
            if isinstance(instruction, bytes):
                instruction = instruction.decode("utf-8")

    return images, actions, instruction


def compute_signals(images, actions, gripper_indices=(6, 13)):
    """Compute raw signals: gripper state, action velocity, pixel MSE."""
    T = len(images)

    # Gripper states
    gripper_left = np.zeros(T)
    gripper_right = np.zeros(T)
    if actions is not None:
        gi_l, gi_r = gripper_indices
        if gi_l < actions.shape[1]:
            gripper_left = actions[:, gi_l]
        if gi_r < actions.shape[1]:
            gripper_right = actions[:, gi_r]

    # Action velocity (norm, excluding gripper dims)
    velocity = np.zeros(T)
    if actions is not None:
        exclude = set(gripper_indices)
        mask = [i for i in range(actions.shape[1]) if i not in exclude]
        for t in range(1, T):
            velocity[t] = np.linalg.norm(actions[t, mask] - actions[t - 1, mask])

    # Pixel MSE between consecutive frames
    pixel_mse = np.zeros(T)
    for t in range(1, T):
        f_curr = images[t].astype(np.float32) / 255.0
        f_prev = images[t - 1].astype(np.float32) / 255.0
        pixel_mse[t] = np.mean((f_curr - f_prev) ** 2)

    return {
        "gripper_left": gripper_left,
        "gripper_right": gripper_right,
        "velocity": velocity,
        "pixel_mse": pixel_mse,
    }


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def vis1_timeline(images, all_keyframes, signals, episode_name, out_path):
    """
    Vis 1: Video strip + keyframe markers per strategy on a shared timeline.
    Top: sampled video frames. Bottom: per-strategy marker rows.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch

    T = len(images)
    n_strategies = len(all_keyframes)

    fig, axes = plt.subplots(
        n_strategies + 1, 1,
        figsize=(16, 2.0 * (n_strategies + 1)),
        gridspec_kw={"height_ratios": [3] + [1] * n_strategies},
    )
    fig.suptitle(f"Vis 1: Timeline — {episode_name}  (T={T})", fontsize=14, y=0.98)

    # Top: sampled video frames
    ax = axes[0]
    sample_step = max(1, T // 12)
    sample_indices = list(range(0, T, sample_step))
    n_samples = len(sample_indices)
    strip = []
    thumb_h, thumb_w = 48, 64
    for idx in sample_indices:
        img = cv2.resize(images[idx], (thumb_w, thumb_h))
        strip.append(img)
    if strip:
        row_img = np.concatenate(strip, axis=1)
        ax.imshow(row_img)
        for i, idx in enumerate(sample_indices):
            ax.text(
                i * thumb_w + thumb_w // 2, thumb_h + 4,
                str(idx), ha="center", va="top", fontsize=6,
            )
    ax.set_xlim(0, n_samples * thumb_w)
    ax.set_yticks([])
    ax.set_ylabel("video", fontsize=9)

    # Per-strategy marker rows
    colors = {"uniform": "#2196F3", "gripper": "#E91E63", "visual": "#4CAF50"}
    for i, (name, kfs) in enumerate(all_keyframes.items()):
        ax = axes[i + 1]
        indices = [kf.frame_index for kf in kfs]
        ax.set_xlim(0, T)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(name, fontsize=9)
        # timeline bar
        ax.axhline(0.5, color="#ccc", linewidth=4, solid_capstyle="round")
        # markers
        c = colors.get(name, "#999")
        for idx in indices:
            ax.plot(idx, 0.5, "v", color=c, markersize=10)
            ax.text(idx, 0.15, str(idx), ha="center", fontsize=6, color=c)
        ax.set_xlabel("frame" if i == n_strategies - 1 else "")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def vis2_sidebyside(all_keyframes, episode_name, out_path):
    """
    Vis 2: Extracted keyframe images side-by-side per strategy.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_strategies = len(all_keyframes)
    max_kf = max(len(kfs) for kfs in all_keyframes.values())
    max_kf = min(max_kf, 10)  # cap columns

    fig, axes = plt.subplots(
        n_strategies, max_kf,
        figsize=(2.2 * max_kf, 2.5 * n_strategies),
    )
    if n_strategies == 1:
        axes = [axes]
    fig.suptitle(f"Vis 2: Extracted Keyframes — {episode_name}", fontsize=13, y=1.0)

    for row, (name, kfs) in enumerate(all_keyframes.items()):
        display_kfs = kfs[:max_kf]
        for col in range(max_kf):
            ax = axes[row][col] if max_kf > 1 else axes[row]
            if col < len(display_kfs):
                img = base64_to_numpy(display_kfs[col].image_b64)
                ax.imshow(img)
                ax.set_title(f"f{display_kfs[col].frame_index}", fontsize=8)
            else:
                ax.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(name, fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def vis3_signals(images, actions, signals, all_keyframes, episode_name, out_path):
    """
    Vis 3: Raw signal curves (gripper, velocity, pixel MSE) with keyframe markers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(images)
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Vis 3: Signals — {episode_name}  (T={T})", fontsize=13)

    # Plot 1: Gripper state
    ax = axes[0]
    ax.plot(signals["gripper_left"], label="gripper_left (action[6])", color="#E91E63")
    ax.plot(signals["gripper_right"], label="gripper_right (action[13])", color="#9C27B0", alpha=0.7)
    ax.set_ylabel("gripper state")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Action velocity
    ax = axes[1]
    ax.plot(signals["velocity"], label="action velocity (excl gripper)", color="#FF9800")
    ax.set_ylabel("velocity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Pixel MSE
    ax = axes[2]
    ax.plot(signals["pixel_mse"], label="pixel MSE (consecutive frames)", color="#4CAF50")
    ax.set_ylabel("pixel MSE")
    ax.set_xlabel("frame")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Overlay keyframe markers on all plots
    colors = {"uniform": "#2196F3", "gripper": "#E91E63", "visual": "#4CAF50"}
    for name, kfs in all_keyframes.items():
        c = colors.get(name, "#999")
        for kf in kfs:
            for ax in axes:
                ax.axvline(kf.frame_index, color=c, alpha=0.3, linewidth=1, linestyle="--")

    # Add legend for vertical lines (bottom plot only)
    for name, c in colors.items():
        axes[2].axvline(-1, color=c, alpha=0.4, linewidth=1, linestyle="--", label=f"{name} kf")
    axes[2].legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def vis4_event_zoom(images, actions, all_keyframes, signals, episode_name, out_path):
    """
    Vis 4: Close-up of detected gripper events (+-3 frames around each event).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = len(images)
    gripper_kfs = all_keyframes.get("gripper", [])
    # Skip first/last (always included), focus on detected events
    event_kfs = [kf for kf in gripper_kfs if 3 < kf.frame_index < T - 3]
    if not event_kfs:
        event_kfs = gripper_kfs[1:4]  # fallback: take a few

    n_events = min(len(event_kfs), 4)
    if n_events == 0:
        print(f"  Skipping vis4 for {episode_name}: no gripper events detected")
        return

    window = 3  # frames before and after
    n_cols = 2 * window + 1

    fig, axes = plt.subplots(
        n_events, n_cols,
        figsize=(2.0 * n_cols, 2.8 * n_events),
    )
    if n_events == 1:
        axes = [axes]
    fig.suptitle(f"Vis 4: Event Zoom — {episode_name}", fontsize=13, y=1.01)

    for row, kf in enumerate(event_kfs[:n_events]):
        center = kf.frame_index
        for col, offset in enumerate(range(-window, window + 1)):
            idx = center + offset
            ax = axes[row][col]
            if 0 <= idx < T:
                img = cv2.resize(images[idx], (128, 96))
                ax.imshow(img)

                # Gripper value annotation
                g_val = ""
                if actions is not None and 6 < actions.shape[1]:
                    g_val = f"g={actions[idx, 6]:.2f}"
                ax.set_xlabel(g_val, fontsize=7)
            else:
                ax.set_visible(False)

            ax.set_xticks([])
            ax.set_yticks([])

            # Highlight center frame
            if offset == 0:
                ax.set_title(f"*f{idx}*", fontsize=9, fontweight="bold", color="red")
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2)
            else:
                ax.set_title(f"f{idx}", fontsize=8)

            if col == 0:
                ax.set_ylabel(f"event {row + 1}", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def vis5_statistics(all_results, out_path):
    """
    Vis 5: Cross-episode aggregate statistics (keyframe count, gap distribution).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strategy_names = list(STRATEGIES.keys())
    episode_names = [r["episode_name"] for r in all_results]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Vis 5: Cross-Episode Keyframe Statistics", fontsize=13)

    # Left: keyframe count per (episode, strategy)
    ax = axes[0]
    x = np.arange(len(episode_names))
    bar_w = 0.25
    colors = {"uniform": "#2196F3", "gripper": "#E91E63", "visual": "#4CAF50"}
    for i, s in enumerate(strategy_names):
        counts = [r["strategies"][s]["n_keyframes"] for r in all_results]
        ax.bar(x + i * bar_w, counts, bar_w, label=s, color=colors.get(s, "#999"))
    ax.set_xticks(x + bar_w)
    ax.set_xticklabels(episode_names, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("# keyframes")
    ax.set_title("Keyframe Count per Strategy")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # Right: inter-keyframe gap distribution (boxplot)
    ax = axes[1]
    gap_data = {s: [] for s in strategy_names}
    for r in all_results:
        for s in strategy_names:
            indices = r["strategies"][s]["frame_indices"]
            gaps = [indices[j + 1] - indices[j] for j in range(len(indices) - 1)]
            gap_data[s].extend(gaps)

    bp_data = [gap_data[s] for s in strategy_names]
    bp = ax.boxplot(bp_data, labels=strategy_names, patch_artist=True)
    for patch, s in zip(bp["boxes"], strategy_names):
        patch.set_facecolor(colors.get(s, "#ccc"))
        patch.set_alpha(0.6)
    ax.set_ylabel("gap (frames)")
    ax.set_title("Inter-Keyframe Gap Distribution")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_episode(hdf5_path, out_dir):
    """Run all strategies on one episode and generate vis 1-4."""
    episode_name = Path(hdf5_path).stem
    print(f"\n{'='*60}")
    print(f"Processing: {episode_name}")
    print(f"  Path: {hdf5_path}")

    # Load data
    images, actions, instruction = load_hdf5_data(str(hdf5_path))
    T = len(images)
    print(f"  Frames: {T}, Instruction: {instruction[:80] if instruction else '(none)'}")

    # Compute raw signals
    signals = compute_signals(images, actions)

    # Extract keyframes with each strategy
    all_keyframes = {}
    result_entry = {
        "episode_name": episode_name,
        "hdf5_path": str(hdf5_path),
        "total_frames": T,
        "instruction": instruction,
        "strategies": {},
    }

    for name, params in STRATEGIES.items():
        print(f"  Strategy: {name} ...", end=" ")
        kfs = extract_keyframes_from_hdf5(
            str(hdf5_path), max_keyframes=MAX_KEYFRAMES, use_cache=False, **params
        )
        all_keyframes[name] = kfs
        indices = [kf.frame_index for kf in kfs]
        result_entry["strategies"][name] = {
            "n_keyframes": len(kfs),
            "frame_indices": indices,
            "params": params,
        }
        print(f"{len(kfs)} keyframes at {indices}")

    # Generate visualizations 1-4
    ep_dir = out_dir / episode_name
    ep_dir.mkdir(parents=True, exist_ok=True)

    vis1_timeline(images, all_keyframes, signals, episode_name, ep_dir / "vis1_timeline.png")
    vis2_sidebyside(all_keyframes, episode_name, ep_dir / "vis2_sidebyside.png")
    vis3_signals(images, actions, signals, all_keyframes, episode_name, ep_dir / "vis3_signals.png")
    vis4_event_zoom(images, actions, all_keyframes, signals, episode_name, ep_dir / "vis4_event_zoom.png")

    return result_entry


def main():
    parser = argparse.ArgumentParser(description="Rule-based keyframe extraction experiment")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="HDF5 data directory")
    parser.add_argument("--episodes", nargs="+", type=str, default=None, help="Specific HDF5 files")
    parser.add_argument("--max-episodes", type=int, default=MAX_EPISODES, help="Max episodes to process")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="List episodes without extracting")
    args = parser.parse_args()

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = SCRIPT_DIR / "results" / "rule"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Find episodes
    if args.episodes:
        episode_paths = [Path(p) for p in args.episodes]
    else:
        episode_paths = find_episodes(args.data_dir, args.max_episodes)

    if not episode_paths:
        print(f"No HDF5 files found in {args.data_dir}")
        print("Usage: python run_keyframe_rule.py --data-dir /path/to/hdf5/")
        print("   or: python run_keyframe_rule.py --episodes ep1.hdf5 ep2.hdf5")
        sys.exit(1)

    print(f"Found {len(episode_paths)} episodes:")
    for p in episode_paths:
        print(f"  {p}")

    if args.dry_run:
        print("\nDry run — exiting without extraction.")
        for p in episode_paths:
            try:
                images, actions, instruction = load_hdf5_data(str(p))
                T = len(images)
                act_shape = actions.shape if actions is not None else "None"
                print(f"  {p.name}: T={T}, actions={act_shape}, instr='{instruction[:60]}'")
            except Exception as e:
                print(f"  {p.name}: ERROR {e}")
        return

    # Process each episode
    all_results = []
    for ep_path in episode_paths:
        try:
            result = process_episode(ep_path, out_dir)
            all_results.append(result)
        except Exception as e:
            print(f"  ERROR processing {ep_path}: {e}")
            import traceback
            traceback.print_exc()

    if not all_results:
        print("No episodes processed successfully.")
        sys.exit(1)

    # Vis 5: Cross-episode statistics
    vis5_statistics(all_results, out_dir / "vis5_statistics.png")

    # Save extraction log
    log_path = out_dir / "extraction_log.json"
    with open(log_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nExtraction log saved: {log_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in all_results:
        print(f"\n{r['episode_name']} (T={r['total_frames']}):")
        for s, info in r["strategies"].items():
            indices = info["frame_indices"]
            gaps = [indices[j + 1] - indices[j] for j in range(len(indices) - 1)]
            max_gap = max(gaps) if gaps else 0
            print(f"  {s:10s}: {info['n_keyframes']:2d} kf, max_gap={max_gap:3d}")

    print(f"\nAll outputs saved to: {out_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
