#!/usr/bin/env python
"""
Visualize cached keyframes to verify if they capture true event frames.

Reads .keyframe_cache/*.json directly, decodes base64 images, and generates
a diagnostic image per cache entry showing:
  - All extracted keyframes in a grid (with frame index labels)
  - Timeline bar showing where keyframes fall in the episode
  - Source metadata (strategy, params, video path)

Usage:
    # Visualize all cached entries
    python visualize_cache.py --cache-dir /path/to/.keyframe_cache

    # Save output images
    python visualize_cache.py --cache-dir /path/to/.keyframe_cache --out-dir ./vis_output

    # Also load original HDF5 to show ±3 frame context around each keyframe
    python visualize_cache.py --cache-dir /path/to/.keyframe_cache --hdf5 /path/to/episode.hdf5
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from base64 import b64decode

import cv2


def b64_to_rgb(image_b64: str) -> np.ndarray:
    """Decode base64 JPEG to RGB numpy array."""
    img_bytes = b64decode(image_b64)
    img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def load_cache_entries(cache_dir: str):
    """Load all cache JSON files, return list of dicts."""
    cache_path = Path(cache_dir)
    entries = []
    for f in sorted(cache_path.glob("*.json")):
        try:
            with open(f, "r") as fp:
                data = json.load(fp)
            data["_cache_file"] = str(f)
            entries.append(data)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Skip {f.name}: {e}")
    return entries


def draw_keyframe_grid(entry, out_path=None, thumb_size=(160, 120), max_cols=6):
    """
    Draw a single image showing all keyframes from one cache entry.

    Layout:
      Title: strategy, video path, num_keyframes
      Row of keyframe thumbnails with frame index labels
      Timeline bar at bottom
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    keyframes = entry.get("keyframes", [])
    strategy = entry.get("strategy", "unknown")
    video_path = entry.get("video_path", "unknown")
    params = entry.get("params", {})
    n_kf = len(keyframes)

    if n_kf == 0:
        print(f"  No keyframes in entry, skipping")
        return

    # Decode images
    images = []
    indices = []
    for kf in keyframes:
        img = b64_to_rgb(kf["image_b64"])
        images.append(img)
        indices.append(kf["frame_index"])

    # Layout
    n_cols = min(n_kf, max_cols)
    n_rows = (n_kf + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(2.8 * n_cols, 2.8 * n_rows + 1.8))

    # Use gridspec: top rows for images, bottom for timeline
    gs = gridspec.GridSpec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [0.3],
                           hspace=0.35, wspace=0.15)

    # Title
    video_name = Path(video_path).name if video_path != "unknown" else "unknown"
    params_str = ", ".join(f"{k}={v}" for k, v in params.items() if k != "use_cache")
    fig.suptitle(
        f"strategy={strategy}  |  {n_kf} keyframes  |  {params_str}\n{video_name}",
        fontsize=10, y=0.98,
    )

    # Keyframe thumbnails
    for i, (img, idx) in enumerate(zip(images, indices)):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.imshow(img)
        ax.set_title(f"f{idx}", fontsize=9, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    # Hide unused cells
    for i in range(n_kf, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        ax = fig.add_subplot(gs[row, col])
        ax.set_visible(False)

    # Timeline bar at bottom
    ax_tl = fig.add_subplot(gs[n_rows, :])
    max_frame = max(indices) if indices else 1
    # Estimate total frames (last keyframe index + some margin)
    total_est = int(max_frame * 1.1) + 5

    ax_tl.set_xlim(0, total_est)
    ax_tl.set_ylim(0, 1)
    ax_tl.axhline(0.5, color="#ddd", linewidth=6, solid_capstyle="round")

    for idx in indices:
        ax_tl.plot(idx, 0.5, "v", color="#E91E63", markersize=8)
        ax_tl.text(idx, 0.1, str(idx), ha="center", fontsize=6, color="#E91E63")

    ax_tl.set_xlabel(f"frame (estimated total ≈ {total_est})", fontsize=8)
    ax_tl.set_yticks([])
    ax_tl.set_title("keyframe positions", fontsize=8)

    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")
    else:
        plt.show()
        plt.close(fig)


def draw_event_context(entry, hdf5_path, out_path=None, window=3):
    """
    For each keyframe, show ±window surrounding frames from the original HDF5
    so you can judge if the keyframe lands on the actual event.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import h5py

    keyframes = entry.get("keyframes", [])
    strategy = entry.get("strategy", "unknown")
    indices = [kf["frame_index"] for kf in keyframes]

    # Skip trivial keyframes (first/last), focus on detected events
    if len(indices) > 2:
        event_indices = indices[1:-1]  # skip first and last
    else:
        event_indices = indices

    # Cap at 5 events
    event_indices = event_indices[:5]
    n_events = len(event_indices)
    if n_events == 0:
        return

    # Load images from HDF5
    with h5py.File(hdf5_path, "r") as f:
        if "observations/unified_image" in f:
            all_images = f["observations/unified_image"][:]
        elif "observations/images/cam_high" in f:
            all_images = f["observations/images/cam_high"][:]
        else:
            print(f"  No image data in {hdf5_path}, skip context vis")
            return

        actions = None
        if "action" in f:
            actions = f["action"][:]
        elif "actions" in f:
            actions = f["actions"][:]

    T = len(all_images)
    n_cols = 2 * window + 1

    fig, axes = plt.subplots(n_events, n_cols, figsize=(2.0 * n_cols, 2.5 * n_events))
    if n_events == 1:
        axes = [axes]

    fig.suptitle(f"Event Context (±{window} frames)  |  strategy={strategy}", fontsize=11, y=1.01)

    for row, center_idx in enumerate(event_indices):
        for col, offset in enumerate(range(-window, window + 1)):
            frame_idx = center_idx + offset
            ax = axes[row][col]

            if 0 <= frame_idx < T:
                img = cv2.resize(all_images[frame_idx], (160, 120))
                ax.imshow(img)

                # Show gripper value if actions available
                label_parts = [f"f{frame_idx}"]
                if actions is not None and actions.shape[1] > 6:
                    label_parts.append(f"g={actions[frame_idx, 6]:.2f}")
                ax.set_xlabel("\n".join(label_parts), fontsize=7)
            else:
                ax.text(0.5, 0.5, "OOB", ha="center", va="center", transform=ax.transAxes)

            ax.set_xticks([])
            ax.set_yticks([])

            # Highlight center
            if offset == 0:
                ax.set_title(f"★ f{frame_idx}", fontsize=9, fontweight="bold", color="red")
                for spine in ax.spines.values():
                    spine.set_edgecolor("red")
                    spine.set_linewidth(2.5)
            else:
                ax.set_title(f"f{frame_idx}", fontsize=8, color="#666")

        # Row label
        axes[row][0].set_ylabel(f"event {row+1}", fontsize=9)

    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")
    else:
        plt.show()
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize cached keyframes")
    parser.add_argument(
        "--cache-dir", type=str, required=True,
        help="Path to .keyframe_cache directory"
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help="Output directory for images (default: display with plt.show)"
    )
    parser.add_argument(
        "--hdf5", type=str, default=None,
        help="Optional: HDF5 file to show ±3 frame context around each event"
    )
    parser.add_argument(
        "--max-entries", type=int, default=10,
        help="Max cache entries to visualize"
    )
    args = parser.parse_args()

    # Load cache
    entries = load_cache_entries(args.cache_dir)
    print(f"Found {len(entries)} cache entries in {args.cache_dir}")

    if not entries:
        print("No cache entries found. Run extract_keyframes_from_hdf5() first.")
        return

    # Output dir
    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = None

    # Print summary of all entries
    print(f"\n{'Strategy':<20s} {'#KF':>4s}  {'Indices':<50s}  {'Source'}")
    print("-" * 100)
    for entry in entries[:args.max_entries]:
        strategy = entry.get("strategy", "?")
        kfs = entry.get("keyframes", [])
        n = len(kfs)
        indices = [kf["frame_index"] for kf in kfs]
        idx_str = str(indices[:12]) + ("..." if len(indices) > 12 else "")
        source = Path(entry.get("video_path", "?")).name
        print(f"{strategy:<20s} {n:>4d}  {idx_str:<50s}  {source}")

    # Visualize each entry
    for i, entry in enumerate(entries[:args.max_entries]):
        strategy = entry.get("strategy", "unknown")
        source_name = Path(entry.get("video_path", "unknown")).stem
        print(f"\n--- Entry {i+1}: {strategy} / {source_name} ---")

        # Grid visualization
        grid_path = None
        if out_dir:
            grid_path = out_dir / f"{i+1:02d}_{source_name}_{strategy}_grid.png"
        draw_keyframe_grid(entry, out_path=grid_path)

        # Context visualization (if HDF5 provided)
        if args.hdf5 and os.path.exists(args.hdf5):
            ctx_path = None
            if out_dir:
                ctx_path = out_dir / f"{i+1:02d}_{source_name}_{strategy}_context.png"
            draw_event_context(entry, args.hdf5, out_path=ctx_path)

    print(f"\nDone. {min(len(entries), args.max_entries)} entries visualized.")
    if out_dir:
        print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
