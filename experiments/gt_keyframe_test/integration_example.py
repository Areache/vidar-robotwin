"""
Integration example: How to use GT keyframes in the eval pipeline.

This shows multiple integration approaches for your evaluation setup.
"""

import os
import sys
from pathlib import Path

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from extract_keyframes import (
    extract_keyframes_uniform,
    extract_keyframes_visual_change,
    get_cache,
    KeyframeInfo
)
from typing import List


# =============================================================================
# APPROACH 1: Direct injection in eval function (Recommended)
# =============================================================================

def eval_with_gt_subgoals(TASK_ENV, model, observation, gt_keyframes: List[KeyframeInfo]):
    """
    Modified eval function that uses GT keyframes instead of model-generated subgoals.

    Copy this to replace/wrap the eval() function in deploy_policy.py
    """
    from policy.AR.deploy_policy import encode_obs
    import torch
    import torchvision
    from base64 import b64encode

    TASK_ENV.step_lim = model.max_steps
    obs = encode_obs(observation)

    model.set_episode_id(TASK_ENV.ep_num)
    if model.task_config.startswith("demo"):
        instruction = TASK_ENV.instruction
        model.set_demo_instruction(instruction)
    else:
        instruction = TASK_ENV.full_instruction
        model.set_instruction(instruction)

    # INJECT GT KEYFRAMES instead of generating subgoals
    if len(model.obs_cache) == 0:
        print(f"Injecting {len(gt_keyframes)} GT keyframes as subgoals")
        model.current_subgoals = [kf.image_b64 for kf in gt_keyframes]
        model.use_libero_subgoal = True
        model.libero_use_direct_model = False  # Disable model-based generation
        print(f"GT keyframe indices: {[kf.frame_index for kf in gt_keyframes]}")

    model.update_obs(obs)
    print(f"Instruction: {model.prompt}")

    actions = model.get_actions()
    action_idx = 0

    while action_idx < len(actions):
        action = actions[action_idx]
        TASK_ENV.take_action(action, action_type='qpos')
        if TASK_ENV.eval_success:
            break
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)
        action_idx += 1
        if action_idx == len(actions):
            actions += model.get_actions()
    model.save_videos()


# =============================================================================
# APPROACH 2: Monkey-patch the model before evaluation
# =============================================================================

def inject_gt_subgoals_into_model(model, gt_keyframes: List[KeyframeInfo]):
    """
    Inject GT keyframes into model before running eval.

    Call this AFTER reset_model() but BEFORE eval().

    Usage in eval_policy.py:
        reset_func(model)
        inject_gt_subgoals_into_model(model, keyframes)  # ADD THIS LINE
        while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
            eval_func(TASK_ENV, model, observation)
            ...
    """
    model.current_subgoals = [kf.image_b64 for kf in gt_keyframes]
    model.use_libero_subgoal = True
    model.libero_use_direct_model = False

    # Store metadata for logging
    model._gt_keyframe_indices = [kf.frame_index for kf in gt_keyframes]

    print(f"Injected {len(gt_keyframes)} GT keyframes")
    return model


# =============================================================================
# APPROACH 3: GTSubgoalEvalRunner - Full wrapper class
# =============================================================================

class GTSubgoalEvalRunner:
    """
    Wrapper class to run evaluation with GT keyframe subgoals.

    Usage:
        runner = GTSubgoalEvalRunner(
            eval_result_dir="/path/to/eval_result/ar/ddp_causal",
            strategy="uniform",
            interval=8
        )

        # In your eval loop:
        for episode_idx in range(test_num):
            video_path = f"{eval_result_dir}/{task}/episode{episode_idx}.mp4"
            keyframes = runner.get_keyframes(task, video_path)

            reset_func(model)
            runner.inject_keyframes(model, keyframes)

            # ... run eval ...
    """

    def __init__(
        self,
        eval_result_dir: str,
        strategy: str = "uniform",
        interval: int = 8,
        threshold: float = 0.05,
        max_keyframes: int = 20,
        cache_dir: str = None
    ):
        self.eval_result_dir = eval_result_dir
        self.strategy = strategy
        self.interval = interval
        self.threshold = threshold
        self.max_keyframes = max_keyframes
        self.cache = get_cache(cache_dir)

    def get_keyframes(self, task: str, episode_idx: int) -> List[KeyframeInfo]:
        """Get keyframes for a specific episode."""
        video_path = os.path.join(
            self.eval_result_dir, task, f"episode{episode_idx}.mp4"
        )

        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            return []

        if self.strategy == "uniform":
            return extract_keyframes_uniform(
                video_path,
                interval=self.interval,
                max_keyframes=self.max_keyframes,
                use_cache=True,
                cache=self.cache
            )
        else:
            return extract_keyframes_visual_change(
                video_path,
                threshold=self.threshold,
                max_keyframes=self.max_keyframes,
                use_cache=True,
                cache=self.cache
            )

    def inject_keyframes(self, model, keyframes: List[KeyframeInfo]):
        """Inject keyframes into model."""
        return inject_gt_subgoals_into_model(model, keyframes)


# =============================================================================
# APPROACH 4: Minimal patch to eval_policy.py
# =============================================================================

MINIMAL_PATCH_EXAMPLE = """
# Add this to the TOP of eval_policy.py:

from experiments.gt_keyframe_test.integration_example import GTSubgoalEvalRunner

# Initialize runner (do this once)
GT_RUNNER = GTSubgoalEvalRunner(
    eval_result_dir="/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar-robotwin/eval_result/ar/ddp_causal",
    strategy="uniform",
    interval=8
)

# Then in eval_policy() function, around line 298-302, change:

# BEFORE:
#     reset_func(model)
#     while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
#         observation = TASK_ENV.get_obs()
#         eval_func(TASK_ENV, model, observation)

# AFTER:
    reset_func(model)

    # Inject GT keyframes
    keyframes = GT_RUNNER.get_keyframes(task_name, TASK_ENV.test_num)
    if keyframes:
        GT_RUNNER.inject_keyframes(model, keyframes)

    while TASK_ENV.take_action_cnt < TASK_ENV.step_lim:
        observation = TASK_ENV.get_obs()
        eval_func(TASK_ENV, model, observation)
"""


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="GT Keyframe Integration Example")
    parser.add_argument("--task", type=str, default="adjust_bottle")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--eval-result-dir", type=str,
                        default="/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar-robotwin/eval_result/ar/ddp_causal")
    args = parser.parse_args()

    print("=" * 60)
    print("GT Keyframe Integration Example")
    print("=" * 60)

    # Initialize runner
    runner = GTSubgoalEvalRunner(
        eval_result_dir=args.eval_result_dir,
        strategy="uniform",
        interval=8,
        max_keyframes=20
    )

    # Get keyframes
    print(f"\nGetting keyframes for task={args.task}, episode={args.episode}")
    keyframes = runner.get_keyframes(args.task, args.episode)

    if keyframes:
        print(f"\nExtracted {len(keyframes)} keyframes:")
        for kf in keyframes[:5]:  # Show first 5
            print(f"  Frame {kf.frame_index} @ {kf.timestamp:.2f}s")
        if len(keyframes) > 5:
            print(f"  ... and {len(keyframes) - 5} more")

        print("\n" + "=" * 60)
        print("To integrate, add this to your eval_policy.py:")
        print("=" * 60)
        print(MINIMAL_PATCH_EXAMPLE)
    else:
        print(f"No video found at: {args.eval_result_dir}/{args.task}/episode{args.episode}.mp4")
        print("\nMake sure the video exists, or check the path.")
