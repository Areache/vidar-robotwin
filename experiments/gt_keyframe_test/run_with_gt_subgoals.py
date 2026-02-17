"""
Test script for running policy with ground truth keyframe subgoals.

This script demonstrates how to:
1. Extract keyframes from evaluation videos
2. Inject them as subgoals into the AR policy
3. Run evaluation with GT subgoals instead of model-generated ones
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import List, Optional

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from extract_keyframes import (
    KeyframeInfo,
    KeyframeCache,
    extract_keyframes_uniform,
    extract_keyframes_visual_change,
    find_eval_videos,
    get_video_info,
    get_cache
)


def inject_gt_subgoals(policy, keyframes: List[KeyframeInfo]):
    """
    Inject GT keyframes into policy WITHOUT disrupting causal structure.

    The key is that subgoals go through subgoal_frames parameter,
    which is processed separately from conditional frames.

    Args:
        policy: AR policy instance
        keyframes: List of KeyframeInfo objects with extracted keyframes

    Returns:
        Modified policy with GT subgoals
    """
    # Set the GT keyframes as subgoals
    policy.current_subgoals = [kf.image_b64 for kf in keyframes]
    policy.use_libero_subgoal = True

    # IMPORTANT: Disable model-based subgoal generation
    # This prevents the policy from overwriting our GT subgoals
    policy.libero_use_direct_model = False

    # Store keyframe metadata for logging
    policy._gt_keyframe_indices = [kf.frame_index for kf in keyframes]
    policy._gt_keyframe_timestamps = [kf.timestamp for kf in keyframes]

    print(f"Injected {len(keyframes)} GT keyframes as subgoals")
    print(f"  Frame indices: {policy._gt_keyframe_indices}")

    return policy


def reset_gt_subgoals(policy):
    """
    Reset GT subgoals for a new episode.

    Call this before starting a new episode to clear any stale subgoals.
    """
    policy.current_subgoals = []
    policy.subgoal_idx = 0
    policy._gt_keyframe_indices = []
    policy._gt_keyframe_timestamps = []


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def extract_keyframes_for_task(
    eval_result_dir: str,
    task: str,
    strategy: str = "uniform",
    interval: int = 8,
    threshold: float = 0.05,
    max_keyframes: int = 20
) -> dict:
    """
    Extract keyframes for all episodes of a task.

    Args:
        eval_result_dir: Base directory for eval results
        task: Task name
        strategy: "uniform" or "visual_change"
        interval: Frame interval for uniform extraction
        threshold: Threshold for visual change detection
        max_keyframes: Maximum keyframes per episode

    Returns:
        Dict mapping episode paths to their keyframes
    """
    videos = find_eval_videos(eval_result_dir, task)
    print(f"Found {len(videos)} videos for task '{task}'")

    episode_keyframes = {}

    for video_path in videos:
        try:
            info = get_video_info(video_path)
            print(f"\nProcessing: {video_path}")
            print(f"  Duration: {info['duration']:.1f}s, "
                  f"Frames: {info['total_frames']}, "
                  f"FPS: {info['fps']:.1f}")

            if strategy == "uniform":
                keyframes = extract_keyframes_uniform(
                    video_path,
                    interval=interval,
                    max_keyframes=max_keyframes
                )
            elif strategy == "visual_change":
                keyframes = extract_keyframes_visual_change(
                    video_path,
                    threshold=threshold,
                    max_keyframes=max_keyframes
                )
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            episode_keyframes[video_path] = keyframes
            print(f"  Extracted {len(keyframes)} keyframes")

        except Exception as e:
            print(f"  Error: {e}")
            continue

    return episode_keyframes


class GTSubgoalEvaluator:
    """
    Evaluator that runs policy with GT keyframe subgoals.

    This class wraps the evaluation process to inject GT subgoals
    extracted from successful demonstration videos.
    """

    def __init__(self, config: dict):
        self.config = config
        self.eval_result_dir = config.get(
            "eval_result_dir",
            "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar-robotwin/eval_result/ar/ddp_causal"
        )
        self.strategy = config.get("extraction_strategy", "uniform")
        self.interval = config.get("keyframe_interval", 8)
        self.threshold = config.get("visual_change_threshold", 0.05)
        self.max_keyframes = config.get("max_keyframes", 20)
        self.tasks = config.get("tasks", ["adjust_bottle"])

        # Set up file-based cache
        self.use_cache = config.get("use_cache", True)
        cache_dir = config.get("cache_dir", None)
        self._cache = get_cache(cache_dir) if self.use_cache else None

    def get_keyframes_for_episode(
        self,
        task: str,
        episode_video_path: str
    ) -> List[KeyframeInfo]:
        """
        Get keyframes for a specific episode.

        Uses file-based caching to avoid re-extracting keyframes.
        Cache is automatically invalidated when source video changes.
        """
        if self.strategy == "uniform":
            keyframes = extract_keyframes_uniform(
                episode_video_path,
                interval=self.interval,
                max_keyframes=self.max_keyframes,
                use_cache=self.use_cache,
                cache=self._cache
            )
        else:
            keyframes = extract_keyframes_visual_change(
                episode_video_path,
                threshold=self.threshold,
                max_keyframes=self.max_keyframes,
                use_cache=self.use_cache,
                cache=self._cache
            )
        return keyframes

    def get_cache_stats(self) -> dict:
        """Get cache statistics."""
        if self._cache:
            return self._cache.get_stats()
        return {"cache_enabled": False}

    def clear_cache(self) -> int:
        """Clear all cached keyframes."""
        if self._cache:
            return self._cache.clear()
        return 0

    def prepare_policy(self, policy, task: str, episode_video_path: str):
        """
        Prepare policy with GT subgoals for an episode.

        Args:
            policy: AR policy instance
            task: Task name
            episode_video_path: Path to the episode video

        Returns:
            Policy with GT subgoals injected
        """
        keyframes = self.get_keyframes_for_episode(task, episode_video_path)
        return inject_gt_subgoals(policy, keyframes)

    def preload_all_keyframes(self):
        """
        Preload keyframes for all configured tasks.

        This extracts and caches keyframes for all episodes in all configured tasks.
        Subsequent calls to get_keyframes_for_episode will use the cached data.
        """
        print(f"Preloading keyframes for {len(self.tasks)} tasks...")
        for task in self.tasks:
            try:
                videos = find_eval_videos(self.eval_result_dir, task)
                print(f"  Task '{task}': found {len(videos)} videos")
                for video_path in videos:
                    # This will extract and cache the keyframes
                    self.get_keyframes_for_episode(task, video_path)
            except Exception as e:
                print(f"  Error preloading keyframes for task '{task}': {e}")

        if self._cache:
            stats = self._cache.get_stats()
            print(f"Preload complete. Cache: {stats['num_entries']} entries, "
                  f"{stats['total_size_mb']:.2f} MB")


def demo_extraction(config_path: str):
    """
    Demo keyframe extraction without running full evaluation.

    Useful for verifying extraction works correctly.
    """
    config = load_config(config_path)

    eval_result_dir = config.get(
        "eval_result_dir",
        "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar-robotwin/eval_result/ar/ddp_causal"
    )
    tasks = config.get("tasks", ["adjust_bottle"])
    strategy = config.get("extraction_strategy", "uniform")
    interval = config.get("keyframe_interval", 8)
    threshold = config.get("visual_change_threshold", 0.05)
    max_keyframes = config.get("max_keyframes", 20)

    print("=" * 60)
    print("GT Keyframe Extraction Demo")
    print("=" * 60)
    print(f"Eval result dir: {eval_result_dir}")
    print(f"Strategy: {strategy}")
    print(f"Tasks: {tasks}")
    print()

    for task in tasks:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print("=" * 60)

        try:
            episode_keyframes = extract_keyframes_for_task(
                eval_result_dir,
                task,
                strategy=strategy,
                interval=interval,
                threshold=threshold,
                max_keyframes=max_keyframes
            )

            print(f"\nSummary for task '{task}':")
            print(f"  Total episodes: {len(episode_keyframes)}")
            total_kfs = sum(len(kfs) for kfs in episode_keyframes.values())
            print(f"  Total keyframes: {total_kfs}")
            if episode_keyframes:
                avg_kfs = total_kfs / len(episode_keyframes)
                print(f"  Average keyframes per episode: {avg_kfs:.1f}")

        except Exception as e:
            print(f"Error processing task '{task}': {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation with GT keyframe subgoals"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run extraction demo without full evaluation"
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Override task from config"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Single video path to extract keyframes from"
    )

    args = parser.parse_args()

    # Handle single video extraction
    if args.video:
        print(f"Extracting keyframes from: {args.video}")
        info = get_video_info(args.video)
        print(f"Video info: {info}")

        keyframes = extract_keyframes_uniform(args.video, interval=8)
        print(f"\nExtracted {len(keyframes)} keyframes:")
        for kf in keyframes:
            print(f"  Frame {kf.frame_index} @ {kf.timestamp:.2f}s")
        return

    # Load config
    config_path = Path(__file__).parent / args.config
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        print("Using default configuration")
        config = {
            "eval_result_dir": "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar-robotwin/eval_result/ar/ddp_causal",
            "extraction_strategy": "uniform",
            "keyframe_interval": 8,
            "max_keyframes": 20,
            "tasks": ["adjust_bottle"]
        }
    else:
        config = load_config(str(config_path))

    # Override task if specified
    if args.task:
        config["tasks"] = [args.task]

    # Run demo or full evaluation
    if args.demo:
        demo_extraction(str(config_path) if config_path.exists() else None)
    else:
        print("Full evaluation with GT subgoals")
        print("=" * 60)
        print()
        print("To run full evaluation, you need to:")
        print("1. Start the Vidar server")
        print("2. Import and instantiate the AR policy")
        print("3. Use GTSubgoalEvaluator to prepare the policy")
        print()
        print("Example usage:")
        print()
        print("  from run_with_gt_subgoals import GTSubgoalEvaluator, inject_gt_subgoals")
        print("  from policy.AR.ar import ARPolicy")
        print()
        print("  # Create evaluator")
        print("  evaluator = GTSubgoalEvaluator(config)")
        print()
        print("  # Get keyframes for an episode")
        print("  keyframes = evaluator.get_keyframes_for_episode(task, video_path)")
        print()
        print("  # Inject into policy")
        print("  inject_gt_subgoals(policy, keyframes)")
        print()
        print("For extraction demo, use: --demo")


if __name__ == "__main__":
    main()
