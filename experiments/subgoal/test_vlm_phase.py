"""
Step 3: VLM Phase Detection Feasibility Test (Local Qwen2.5-VL-7B)

Tests whether Qwen2.5-VL-7B can reliably detect robot task phases from
single observation frames. This validates the core assumption of
Method B (VLM + GT Retrieval).

Tested capabilities:
  1. Task decomposition: break task into subtask phases
  2. Phase detection: identify current phase from observation image
  3. Completion detection: determine if a subtask is complete
  4. Interaction success detection: compare plan video vs actual execution

Usage:
  # Test 1-3: Phase detection on a single video
  python test_vlm_phase.py --video /path/to/episode0.mp4 --task adjust_bottle

  # Test 1-3: All episodes for a task
  python test_vlm_phase.py --task adjust_bottle \
    --eval_dir /mnt/shared-storage-user/.../eval_result/ar/ddp_causal

  # Test 4: Plan vs Actual comparison
  python test_vlm_phase.py --task adjust_bottle \
    --plan_video /path/to/episode0_pred_64.mp4 \
    --actual_video /path/to/episode0.mp4

  # Custom model path
  python test_vlm_phase.py --video /path/to/ep.mp4 --task adjust_bottle \
    --model_path /path/to/Qwen2.5-VL-7B-Instruct
"""

import os
import sys
import json
import argparse
import glob
import time
from pathlib import Path
from typing import List, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
sys.path.insert(0, "/mnt/shared-storage-user/kangli/workspace/cyujie/code/vidar/wan/utils")
from qwen_vl_utils import process_vision_info

DEFAULT_MODEL_PATH = "/mnt/shared-storage-user/kangli/workspace/cyujie/mounts/ailab-public-shared/models--Qwen--Qwen2.5-VL-7B-Instruct"


# ─── VLM Client (Local Qwen2.5-VL) ───────────────────────────────

class VLMClient:
    """Local Qwen2.5-VL-7B inference client."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        print(f"Loading Qwen2.5-VL-7B from {model_path} ...")
        t0 = time.time()
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        print(f"Model loaded in {time.time() - t0:.1f}s")
        self.call_count = 0

    def _generate(self, messages: List[Dict], max_tokens: int = 1024) -> str:
        """Run inference on a chat message list."""
        self.call_count += 1
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)
        # trim input tokens
        generated_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        output = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        return output

    def query_with_image(self, prompt: str, pil_image: Image.Image,
                         system_prompt: str = "") -> str:
        """Send text + single PIL image to VLM."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        messages.append({
            "role": "user",
            "content": [
                {"type": "image", "image": pil_image},
                {"type": "text", "text": prompt},
            ],
        })
        return self._generate(messages)

    def query_with_images(self, prompt: str, pil_images: List[Image.Image],
                          system_prompt: str = "") -> str:
        """Send text + multiple PIL images to VLM."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": system_prompt}]})
        content = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
        return self._generate(messages)


# ─── Frame Extraction ─────────────────────────────────────────────

def extract_frames(video_path: str, frame_indices: List[int]) -> List[Image.Image]:
    """Extract specific frames from a video as PIL Images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    for idx in frame_indices:
        if idx >= total_frames:
            print(f"  Warning: frame {idx} exceeds total {total_frames}, skipping")
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # BGR -> RGB -> PIL
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        else:
            print(f"  Warning: failed to read frame {idx}")

    cap.release()
    return frames


def get_video_total_frames(video_path: str) -> int:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


# ─── Task Descriptions ────────────────────────────────────────────

def load_task_description(task_name: str) -> str:
    """Load task description from task_instruction JSON."""
    json_path = Path(__file__).parent.parent.parent / "description" / "task_instruction" / f"{task_name}.json"
    if json_path.exists():
        with open(json_path) as f:
            data = json.load(f)
        return data.get("full_description", task_name)
    return task_name


def parse_json_response(response: str) -> Dict:
    """Try to parse JSON from VLM response, handling markdown blocks."""
    try:
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw_response": response, "parse_error": True}


# ─── Test 1: Task Decomposition ──────────────────────────────────

def test_task_decomposition(vlm: VLMClient, task_name: str,
                            first_frame: Image.Image) -> Dict:
    """Can VLM decompose a robot task into subtask phases?"""
    task_desc = load_task_description(task_name)

    prompt = f"""You are observing a dual-arm robot (ALOHA-style) on a tabletop.
The robot's task is: "{task_desc}"

Looking at the initial observation image, decompose this task into sequential subtask phases.
For each phase, provide:
1. A short name (e.g., "approach_bottle", "grasp", "lift")
2. A description of what happens
3. Visual completion criteria (what should the image look like when this phase is done)

Output as JSON:
{{
  "task": "{task_name}",
  "phases": [
    {{
      "phase_id": 0,
      "name": "phase_name",
      "description": "what happens",
      "completion_criteria": "visual description of completion"
    }}
  ]
}}"""

    t0 = time.time()
    response = vlm.query_with_image(prompt, first_frame,
        system_prompt="You are a robot task analysis expert. Output valid JSON only.")
    latency = time.time() - t0

    result = parse_json_response(response)
    return {
        "test": "task_decomposition",
        "task": task_name,
        "task_description": task_desc,
        "latency_s": round(latency, 2),
        "result": result,
    }


# ─── Test 2: Phase Detection ─────────────────────────────────────

def test_phase_detection(vlm: VLMClient, task_name: str,
                         phases: List[Dict],
                         frame: Image.Image, frame_idx: int) -> Dict:
    """Given known phases, can VLM identify which phase the robot is in?"""
    phase_list = "\n".join(
        f"  Phase {p['phase_id']}: {p['name']} — {p['description']}"
        for p in phases
    )

    prompt = f"""You are observing a dual-arm robot performing the task "{task_name}".

The task has these phases:
{phase_list}

Looking at the current observation image (frame {frame_idx}):
1. Which phase is the robot currently in?
2. How confident are you? (high/medium/low)
3. Is this phase complete, in progress, or not yet started?

Output as JSON:
{{
  "current_phase_id": <int>,
  "phase_name": "<name>",
  "confidence": "<high|medium|low>",
  "phase_status": "<complete|in_progress|not_started>",
  "reasoning": "<brief explanation>"
}}"""

    t0 = time.time()
    response = vlm.query_with_image(prompt, frame,
        system_prompt="You are a robot task analysis expert. Output valid JSON only.")
    latency = time.time() - t0

    result = parse_json_response(response)
    return {
        "test": "phase_detection",
        "frame_index": frame_idx,
        "latency_s": round(latency, 2),
        "result": result,
    }


# ─── Test 3: Completion Detection ────────────────────────────────

def test_completion_detection(vlm: VLMClient, task_name: str,
                              early_frame: Image.Image, late_frame: Image.Image,
                              early_idx: int, late_idx: int) -> Dict:
    """Can VLM distinguish early vs late task stages?"""
    task_desc = load_task_description(task_name)

    prompt = f"""You are observing a dual-arm robot performing: "{task_desc}"

I'm showing you two observation images from the same episode:
- Image 1: frame {early_idx} (earlier in the episode)
- Image 2: frame {late_idx} (later in the episode)

For each image, estimate the task completion progress (0-100%).
Also describe what you see the robot doing in each image.

Output as JSON:
{{
  "image1": {{
    "frame": {early_idx},
    "progress_pct": <0-100>,
    "description": "<what the robot is doing>"
  }},
  "image2": {{
    "frame": {late_idx},
    "progress_pct": <0-100>,
    "description": "<what the robot is doing>"
  }},
  "progress_order_correct": true
}}"""

    t0 = time.time()
    response = vlm.query_with_images(prompt, [early_frame, late_frame],
        system_prompt="You are a robot task analysis expert. Output valid JSON only.")
    latency = time.time() - t0

    result = parse_json_response(response)
    return {
        "test": "completion_detection",
        "early_frame": early_idx,
        "late_frame": late_idx,
        "latency_s": round(latency, 2),
        "result": result,
    }


# ─── Test 4: Interaction Success Detection (Plan vs Actual) ──────

def test_interaction_success(
    vlm: VLMClient, task_name: str,
    plan_frame: Image.Image, actual_frame: Image.Image,
    frame_idx: int, phase_desc: str = "",
) -> Dict:
    """
    Can VLM determine if the actual execution matched the plan?

    Compares a frame from the video model prediction (plan) with the
    corresponding frame from the actual simulation to detect:
    - Did the robot successfully interact with the object?
    - Where did plan and reality diverge?
    """
    task_desc = load_task_description(task_name)

    prompt = f"""You are a robot execution monitor for the task: "{task_desc}"
{f'Current subtask: {phase_desc}' if phase_desc else ''}

I'm showing you two images from frame {frame_idx}:
- Image 1: PLANNED execution (video model prediction — what SHOULD happen)
- Image 2: ACTUAL execution (simulation — what REALLY happened)

Analyze the differences:
1. Did the robot successfully interact with the target object in the ACTUAL execution?
2. Does the actual state match the planned state?
3. If there is a mismatch, what went wrong?

Output as JSON:
{{
  "interaction_success": <true|false>,
  "plan_actual_match": <true|false>,
  "plan_description": "<what the plan shows the robot doing>",
  "actual_description": "<what actually happened>",
  "mismatch_type": "<none|position_error|grasp_failure|object_missed|timing_error|other>",
  "mismatch_detail": "<specific description if mismatch>",
  "confidence": "<high|medium|low>",
  "should_retry": <true|false>
}}"""

    t0 = time.time()
    response = vlm.query_with_images(prompt, [plan_frame, actual_frame],
        system_prompt="You are a robot execution monitor. Compare planned vs actual execution. Output valid JSON only.")
    latency = time.time() - t0

    result = parse_json_response(response)
    return {
        "test": "interaction_success",
        "frame_index": frame_idx,
        "latency_s": round(latency, 2),
        "result": result,
    }


def run_interaction_test(
    plan_video: str, actual_video: str,
    task_name: str, vlm: VLMClient,
    sample_count: int = 8,
) -> Dict:
    """
    Run Test 4: Compare plan video vs actual video at matched frames.

    Args:
        plan_video: Path to video model prediction (episode*_pred_*.mp4)
        actual_video: Path to actual simulation result (episode*.mp4)
    """
    print(f"\n{'='*60}")
    print(f"Test 4: Interaction Success Detection (Plan vs Actual)")
    print(f"  Plan:   {plan_video}")
    print(f"  Actual: {actual_video}")
    print(f"  Task:   {task_name}")
    print(f"{'='*60}\n")

    plan_total = get_video_total_frames(plan_video)
    actual_total = get_video_total_frames(actual_video)
    print(f"Plan frames: {plan_total}, Actual frames: {actual_total}")

    # Sample at matched normalized positions (handles different lengths)
    plan_indices = np.linspace(0, plan_total - 1, sample_count, dtype=int).tolist()
    actual_indices = np.linspace(0, actual_total - 1, sample_count, dtype=int).tolist()
    print(f"Plan sample:   {plan_indices}")
    print(f"Actual sample: {actual_indices}\n")

    plan_frames = extract_frames(plan_video, plan_indices)
    actual_frames = extract_frames(actual_video, actual_indices)

    # Ensure same number of frames
    n = min(len(plan_frames), len(actual_frames))
    plan_frames = plan_frames[:n]
    actual_frames = actual_frames[:n]

    # ── First get task decomposition for phase context ──
    print("Getting task phases for context...")
    decomp = test_task_decomposition(vlm, task_name, actual_frames[0])
    phases = decomp.get("result", {}).get("phases", [])
    if phases:
        print(f"  {len(phases)} phases detected")

    # ── Run interaction comparison at each sample point ──
    interaction_results = []
    success_count = 0
    match_count = 0

    for i in range(n):
        plan_idx = plan_indices[i]
        actual_idx = actual_indices[i]
        progress_pct = int(100 * i / (n - 1)) if n > 1 else 0

        # Find which phase this frame belongs to
        phase_desc = ""
        if phases:
            # Estimate phase from progress
            phase_id = min(int(progress_pct / 100 * len(phases)), len(phases) - 1)
            p = phases[phase_id]
            phase_desc = f"{p.get('name', '')}: {p.get('description', '')}"

        print(f"  [{progress_pct:3d}%] plan_f={plan_idx}, actual_f={actual_idx}...",
              end=" ", flush=True)

        result = test_interaction_success(
            vlm, task_name,
            plan_frames[i], actual_frames[i],
            actual_idx, phase_desc
        )
        interaction_results.append(result)

        r = result.get("result", {})
        success = r.get("interaction_success", "?")
        match = r.get("plan_actual_match", "?")
        mismatch = r.get("mismatch_type", "none")

        if success is True:
            success_count += 1
        if match is True:
            match_count += 1

        status = "OK" if success else f"FAIL({mismatch})"
        print(f"{status}, match={match}, conf={r.get('confidence', '?')} "
              f"({result['latency_s']}s)")

    # ── Summary ──
    total_latency = sum(r["latency_s"] for r in interaction_results)
    print(f"\n{'='*60}")
    print(f"Interaction Test Summary")
    print(f"  Frames compared: {n}")
    print(f"  Interaction success: {success_count}/{n} ({100*success_count/n:.0f}%)")
    print(f"  Plan-actual match:  {match_count}/{n} ({100*match_count/n:.0f}%)")
    print(f"  Total latency: {total_latency:.1f}s")
    print(f"  Avg per comparison: {total_latency/n:.2f}s")

    # Collect mismatch types
    mismatch_types = [
        r.get("result", {}).get("mismatch_type", "unknown")
        for r in interaction_results
        if r.get("result", {}).get("plan_actual_match") is False
    ]
    if mismatch_types:
        from collections import Counter
        print(f"  Mismatch types: {dict(Counter(mismatch_types))}")

    return {
        "test": "interaction_success_batch",
        "plan_video": plan_video,
        "actual_video": actual_video,
        "task": task_name,
        "plan_frames": plan_total,
        "actual_frames": actual_total,
        "num_compared": n,
        "interaction_success_rate": round(success_count / n, 3) if n else 0,
        "plan_actual_match_rate": round(match_count / n, 3) if n else 0,
        "total_latency_s": round(total_latency, 2),
        "results": interaction_results,
    }


# ─── Main Pipeline ────────────────────────────────────────────────

def run_vlm_test(video_path: str, task_name: str, vlm: VLMClient,
                 sample_count: int = 8) -> Dict:
    """Run all three VLM tests on a single episode video."""
    print(f"\n{'='*60}")
    print(f"VLM Phase Detection Test")
    print(f"  Video: {video_path}")
    print(f"  Task:  {task_name}")
    print(f"{'='*60}\n")

    total_frames = get_video_total_frames(video_path)
    print(f"Total frames: {total_frames}")

    sample_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int).tolist()
    print(f"Sampling frames: {sample_indices}\n")

    all_frames = extract_frames(video_path, sample_indices)

    results = {"video": video_path, "task": task_name, "total_frames": total_frames}

    # ── Test 1: Task Decomposition ──
    print("Test 1: Task Decomposition...")
    decomp = test_task_decomposition(vlm, task_name, all_frames[0])
    results["task_decomposition"] = decomp
    phases = decomp.get("result", {}).get("phases", [])

    if phases:
        print(f"  Found {len(phases)} phases ({decomp['latency_s']}s):")
        for p in phases:
            print(f"    [{p.get('phase_id', '?')}] {p.get('name', '?')}: {p.get('description', '?')}")
    else:
        print("  WARNING: No phases detected, using fallback phases")
        phases = [
            {"phase_id": 0, "name": "approach", "description": "Move arm toward object"},
            {"phase_id": 1, "name": "grasp", "description": "Close gripper on object"},
            {"phase_id": 2, "name": "manipulate", "description": "Perform task action"},
            {"phase_id": 3, "name": "complete", "description": "Task finished"},
        ]

    # ── Test 2: Phase Detection on sampled frames ──
    print(f"\nTest 2: Phase Detection on {len(all_frames)} frames...")
    phase_results = []
    for i, (idx, frame) in enumerate(zip(sample_indices, all_frames)):
        print(f"  Frame {idx} ({i+1}/{len(all_frames)})...", end=" ", flush=True)
        pd = test_phase_detection(vlm, task_name, phases, frame, idx)
        phase_results.append(pd)
        r = pd.get("result", {})
        print(f"phase={r.get('current_phase_id', '?')}, "
              f"status={r.get('phase_status', '?')}, "
              f"conf={r.get('confidence', '?')} "
              f"({pd['latency_s']}s)")
    results["phase_detection"] = phase_results

    # ── Test 3: Completion Detection ──
    print(f"\nTest 3: Completion Detection (early vs late)...")
    comp = test_completion_detection(
        vlm, task_name,
        all_frames[0], all_frames[-1],
        sample_indices[0], sample_indices[-1]
    )
    results["completion_detection"] = comp
    r = comp.get("result", {})
    print(f"  Early progress: {r.get('image1', {}).get('progress_pct', '?')}%")
    print(f"  Late progress:  {r.get('image2', {}).get('progress_pct', '?')}%")
    print(f"  Order correct:  {r.get('progress_order_correct', '?')}")
    print(f"  Latency: {comp['latency_s']}s")

    # ── Summary ──
    print(f"\n{'='*60}")
    print(f"Summary: {vlm.call_count} VLM calls total")

    total_latency = (decomp["latency_s"]
                     + sum(pr["latency_s"] for pr in phase_results)
                     + comp["latency_s"])
    print(f"Total inference time: {total_latency:.1f}s")
    print(f"Avg per call: {total_latency / vlm.call_count:.2f}s")

    detected_phases = [
        pr.get("result", {}).get("current_phase_id", -1)
        for pr in phase_results
        if not pr.get("result", {}).get("parse_error", False)
    ]
    if detected_phases:
        is_monotonic = all(a <= b for a, b in zip(detected_phases, detected_phases[1:]))
        print(f"Phase progression monotonic: {is_monotonic}")
        print(f"Detected phase sequence: {detected_phases}")
        results["phase_monotonic"] = is_monotonic

    results["total_latency_s"] = round(total_latency, 2)
    return results


def main():
    parser = argparse.ArgumentParser(description="VLM Phase Detection Feasibility Test (Local Qwen2.5-VL-7B)")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a single GT demo video")
    parser.add_argument("--task", type=str, required=True,
                        help="Task name (e.g., adjust_bottle)")
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="Eval result directory (to test all episodes)")
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to Qwen2.5-VL-7B-Instruct checkpoint")
    parser.add_argument("--sample_count", type=int, default=8,
                        help="Number of frames to sample per episode")
    parser.add_argument("--max_episodes", type=int, default=3,
                        help="Max episodes to test (when using --eval_dir)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: vlm_test_<task>.json)")
    # Test 4: Plan vs Actual comparison
    parser.add_argument("--plan_video", type=str, default=None,
                        help="Path to video model prediction (for Test 4)")
    parser.add_argument("--actual_video", type=str, default=None,
                        help="Path to actual simulation video (for Test 4)")
    args = parser.parse_args()

    # Load model once, reuse across episodes
    vlm = VLMClient(model_path=args.model_path)

    # ── Test 4 mode: Plan vs Actual comparison ──
    if args.plan_video and args.actual_video:
        result = run_interaction_test(
            args.plan_video, args.actual_video,
            args.task, vlm, args.sample_count
        )
        all_results = [result]
    elif args.video:
        results = run_vlm_test(args.video, args.task, vlm, args.sample_count)
        all_results = [results]
    elif args.eval_dir:
        task_dir = os.path.join(args.eval_dir, args.task)
        videos = sorted(glob.glob(os.path.join(task_dir, "episode*.mp4")))
        if not videos:
            print(f"No videos found in {task_dir}")
            sys.exit(1)
        videos = videos[:args.max_episodes]
        print(f"Found {len(videos)} episodes, testing {len(videos)}")

        all_results = []
        for vp in videos:
            results = run_vlm_test(vp, args.task, vlm, args.sample_count)
            all_results.append(results)
    else:
        parser.error("Either --video, --eval_dir, or --plan_video+--actual_video is required")

    output_path = args.output or f"vlm_test_{args.task}.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
