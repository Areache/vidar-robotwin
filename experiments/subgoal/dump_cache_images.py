#!/usr/bin/env python
"""
Dump cached keyframes + original video frames as human-readable image files.

Converts .keyframe_cache/*.json base64 data into actual JPG images,
AND extracts every frame from the original source (HDF5 or MP4) so
you can compare keyframes against the full video.

Usage:
    python dump_cache_images.py --cache-dir /path/to/.keyframe_cache --out-dir ./keyframe_images

Output:
    keyframe_images/
    ├── 01_episode_000001_hdf5_uniform/
    │   ├── keyframes/
    │   │   ├── f0000.jpg          ← extracted keyframe
    │   │   ├── f0008.jpg
    │   │   └── f0016.jpg
    │   ├── original/
    │   │   ├── f0000.jpg          ← ALL original frames (every frame)
    │   │   ├── f0001.jpg
    │   │   ├── ...
    │   │   └── f0080.jpg
    │   └── _info.txt
    └── summary.txt
"""

import os
import sys
import json
import argparse
from pathlib import Path
from base64 import b64decode


def dump_original_from_hdf5(hdf5_path, out_folder, sample_step=1):
    """Extract frames from HDF5 and save as JPGs."""
    import h5py
    import cv2
    import numpy as np

    with h5py.File(hdf5_path, "r") as f:
        if "observations/unified_image" in f:
            images = f["observations/unified_image"]
        elif "observations/images/cam_high" in f:
            images = f["observations/images/cam_high"]
        else:
            print(f"    No image data in {hdf5_path}")
            return 0

        T = images.shape[0]
        count = 0
        for t in range(0, T, sample_step):
            img = images[t]
            # RGB → BGR for cv2 encoding
            if img.ndim == 3 and img.shape[2] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img
            img_path = out_folder / f"f{t:04d}.jpg"
            cv2.imwrite(str(img_path), img_bgr)
            count += 1

    return count


def dump_hdf5_as_video(hdf5_path, out_video_path, fps=10, keyframe_indices=None):
    """Save HDF5 frames as H.264 MP4 video, with keyframe indices marked in red."""
    import h5py
    import cv2
    import numpy as np
    import subprocess

    with h5py.File(hdf5_path, "r") as f:
        if "observations/unified_image" in f:
            images = f["observations/unified_image"]
        elif "observations/images/cam_high" in f:
            images = f["observations/images/cam_high"]
        else:
            print(f"    No image data in {hdf5_path}")
            return 0

        T = images.shape[0]
        h, w = images.shape[1], images.shape[2]

        # Write raw mp4v first
        tmp_path = str(out_video_path) + ".tmp.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (w, h))

        kf_set = set(keyframe_indices or [])
        for t in range(T):
            img = images[t]
            if img.ndim == 3 and img.shape[2] == 3:
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_bgr = img.copy()

            # Draw frame number
            cv2.putText(img_bgr, f"f{t:04d}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Mark keyframes with red border + label
            if t in kf_set:
                cv2.rectangle(img_bgr, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
                cv2.putText(img_bgr, "KEYFRAME", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            writer.write(img_bgr)

        writer.release()

    # Re-encode to H.264 for browser/IDE compatibility
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264",
             "-pix_fmt", "yuv420p", "-crf", "23", str(out_video_path)],
            check=True, capture_output=True
        )
        os.remove(tmp_path)
    except (FileNotFoundError, subprocess.CalledProcessError):
        # ffmpeg not available, fall back to mp4v
        os.rename(tmp_path, str(out_video_path))
        print("    (ffmpeg not found, using mp4v codec — may not play in IDE)")

    return T


def dump_original_from_video(video_path, out_folder, sample_step=1):
    """Extract frames from MP4 video and save as JPGs."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Cannot open video: {video_path}")
        return 0

    count = 0
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_step == 0:
            img_path = out_folder / f"f{frame_idx:04d}.jpg"
            cv2.imwrite(str(img_path), frame)
            count += 1
        frame_idx += 1

    cap.release()
    return count


def main():
    parser = argparse.ArgumentParser(description="Dump cached keyframes + original video as image files")
    parser.add_argument("--cache-dir", type=str, required=True, help="Path to .keyframe_cache/")
    parser.add_argument("--out-dir", type=str, default="./keyframe_images", help="Output directory")
    parser.add_argument(
        "--sample-step", type=int, default=1,
        help="Sample every N-th frame from original (1=all frames, 5=every 5th)"
    )
    parser.add_argument("--skip-original", action="store_true", help="Only dump keyframes, skip original video")
    parser.add_argument("--video", action="store_true", help="Save original frames as MP4 video with keyframes marked")
    parser.add_argument("--fps", type=int, default=10, help="FPS for video output (default: 10)")
    args = parser.parse_args()

    cache_path = Path(args.cache_dir)
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(cache_path.glob("*.json"))
    print(f"Found {len(json_files)} cache entries in {cache_path}")

    if not json_files:
        print("No cache files found.")
        return

    # Track which sources we've already dumped (avoid re-extracting for multiple strategies)
    dumped_sources = {}
    summary_lines = []

    for i, jf in enumerate(json_files):
        with open(jf, "r") as f:
            data = json.load(f)

        strategy = data.get("strategy", "unknown")
        video_path = data.get("video_path", "unknown")
        params = data.get("params", {})
        keyframes = data.get("keyframes", [])

        source_name = Path(video_path).stem
        folder_name = f"{i+1:02d}_{source_name}_{strategy}"
        folder = out_path / folder_name

        # --- Dump keyframes ---
        kf_folder = folder / "keyframes"
        kf_folder.mkdir(parents=True, exist_ok=True)

        indices = []
        for kf in keyframes:
            frame_idx = kf["frame_index"]
            indices.append(frame_idx)
            img_bytes = b64decode(kf["image_b64"])
            img_path = kf_folder / f"f{frame_idx:04d}.jpg"
            with open(img_path, "wb") as img_f:
                img_f.write(img_bytes)

        # --- Save as video (with keyframes marked) ---
        if args.video and video_path != "unknown" and os.path.exists(video_path) and video_path.endswith(".hdf5"):
            video_out = folder / "original.mp4"
            video_out.parent.mkdir(parents=True, exist_ok=True)
            n_frames = dump_hdf5_as_video(video_path, video_out, fps=args.fps, keyframe_indices=indices)
            print(f"    Video saved: {video_out} ({n_frames} frames)")

        # --- Dump original video frames ---
        orig_folder = folder / "original"
        orig_count = 0
        if not args.skip_original and video_path != "unknown" and os.path.exists(video_path):
            # Check if we already dumped this source for another strategy
            if video_path in dumped_sources:
                # Symlink or copy the reference
                ref = dumped_sources[video_path]
                if not orig_folder.exists():
                    try:
                        orig_folder.symlink_to(ref, target_is_directory=True)
                        orig_count = -1  # symlinked
                    except OSError:
                        # Symlink failed (cross-device etc), skip
                        pass
            else:
                orig_folder.mkdir(parents=True, exist_ok=True)
                if video_path.endswith(".hdf5"):
                    orig_count = dump_original_from_hdf5(video_path, orig_folder, args.sample_step)
                elif video_path.endswith((".mp4", ".avi", ".mov")):
                    orig_count = dump_original_from_video(video_path, orig_folder, args.sample_step)
                else:
                    # Try HDF5 first, then video
                    try:
                        orig_count = dump_original_from_hdf5(video_path, orig_folder, args.sample_step)
                    except Exception:
                        try:
                            orig_count = dump_original_from_video(video_path, orig_folder, args.sample_step)
                        except Exception:
                            pass

                if orig_count > 0:
                    dumped_sources[video_path] = orig_folder.resolve()

        elif not args.skip_original and video_path != "unknown":
            print(f"    Source not found: {video_path}")

        # --- Write info file ---
        orig_status = f"{orig_count} frames" if orig_count > 0 else ("symlinked" if orig_count == -1 else "skipped")
        info_text = (
            f"strategy: {strategy}\n"
            f"source: {video_path}\n"
            f"source_exists: {os.path.exists(video_path) if video_path != 'unknown' else False}\n"
            f"params: {json.dumps(params)}\n"
            f"num_keyframes: {len(keyframes)}\n"
            f"frame_indices: {indices}\n"
            f"original_frames: {orig_status}\n"
            f"\n"
            f"HOW TO COMPARE:\n"
            f"  keyframes/ contains ONLY the detected event frames\n"
            f"  original/  contains ALL frames from the source video\n"
            f"  Compare: look at original/f00XX.jpg around each keyframe index\n"
            f"  Example: if keyframe is at f0032, check original/f0030..f0034\n"
        )
        with open(folder / "_info.txt", "w") as f:
            f.write(info_text)

        line = f"{folder_name}: {len(keyframes)} kf at {indices}, original={orig_status}"
        print(f"  {line}")
        summary_lines.append(line)

    # --- Write summary ---
    with open(out_path / "summary.txt", "w") as f:
        f.write(f"Cache: {cache_path}\n")
        f.write(f"Entries: {len(json_files)}\n")
        f.write(f"Sample step: {args.sample_step}\n\n")
        for line in summary_lines:
            f.write(line + "\n")
        f.write(f"\nHOW TO USE:\n")
        f.write(f"  Each folder has keyframes/ and original/ subfolders.\n")
        f.write(f"  Browse original/ to see the full video as images.\n")
        f.write(f"  Keyframe file names = frame indices (f0032.jpg = frame 32).\n")
        f.write(f"  Compare: is keyframes/f0032.jpg at an actual event (grasp/release)?\n")
        f.write(f"  Check surrounding frames: original/f0030.jpg .. original/f0034.jpg\n")

    print(f"\nDone. Images saved to: {out_path}")
    print(f"Each folder contains:")
    print(f"  keyframes/  ← detected event frames")
    print(f"  original/   ← full video as images (every {args.sample_step} frame(s))")
    print(f"  _info.txt   ← metadata + how to compare")


if __name__ == "__main__":
    main()
