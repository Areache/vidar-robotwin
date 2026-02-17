"""
Generate subgoal images using Wan I2V model.

This script:
1. Loads the first frame from a video/HDF5/image
2. Uses predefined subgoal descriptions (from LLM)
3. Generates subgoal images using Wan text-image-to-video model
4. Extracts and saves the target frames

Usage:
    python generate_subgoals.py --first-frame first_frame.jpg
    python generate_subgoals.py --video episode.mp4
    python generate_subgoals.py --hdf5 episode.hdf5
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image

# Add vidar to path
SCRIPT_DIR = Path(__file__).parent.resolve()
VIDAR_ROOT = SCRIPT_DIR.parents[2]  # vidar-robotwin
VIDAR_PATH = VIDAR_ROOT.parent / "vidar"
sys.path.insert(0, str(VIDAR_PATH))
sys.path.insert(0, str(VIDAR_ROOT))


# Default subgoals for stack_bowls_two task (from LLM output)
DEFAULT_SUBGOALS = [
    {
        "subgoal_id": 1,
        "subgoal_name": "bowl_A_grasped",
        "semantic_image_description": "The robot gripper is holding bowl {A}, lifted above the table surface. Bowl {B} remains on the table in its original position.",
        "key_conditions": [
            "Bowl {A} is gripped by the robot",
            "Bowl {A} is elevated from the table",
            "Bowl {B} is stationary on the table"
        ]
    },
    {
        "subgoal_id": 2,
        "subgoal_name": "bowl_A_positioned_above_B",
        "semantic_image_description": "Bowl {A} is held by the robot gripper, positioned directly above bowl {B}. Both bowls are aligned vertically.",
        "key_conditions": [
            "Bowl {A} is above bowl {B}",
            "Bowls are vertically aligned",
            "Bowl {B} is on the table"
        ]
    },
    {
        "subgoal_id": 3,
        "subgoal_name": "bowl_A_stacked_on_B",
        "semantic_image_description": "Bowl {A} is resting on top of bowl {B}, forming a stable stack. The robot gripper has released bowl {A}.",
        "key_conditions": [
            "Bowl {A} is on top of bowl {B}",
            "Stack is stable",
            "Robot gripper is open/released"
        ]
    }
]


def load_first_frame(
    first_frame_path: Optional[str] = None,
    video_path: Optional[str] = None,
    hdf5_path: Optional[str] = None
) -> np.ndarray:
    """
    Load first frame from various sources.

    Returns:
        RGB image as numpy array (H, W, 3)
    """
    if first_frame_path and os.path.exists(first_frame_path):
        img = cv2.imread(first_frame_path)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if video_path and os.path.exists(video_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        raise ValueError(f"Could not read frame from video: {video_path}")

    if hdf5_path and os.path.exists(hdf5_path):
        import h5py
        with h5py.File(hdf5_path, 'r') as f:
            if 'observations/unified_image' in f:
                return f['observations/unified_image'][0]
            elif 'observations/images/cam_high' in f:
                return f['observations/images/cam_high'][0]
        raise ValueError(f"No image data found in HDF5: {hdf5_path}")

    raise ValueError("No valid input source provided")


def create_placeholder_image(size=(640, 480), text="First Frame Placeholder"):
    """Create a placeholder image when no input is available."""
    img = np.ones((size[1], size[0], 3), dtype=np.uint8) * 200
    cv2.putText(img, text, (50, size[1]//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
    return img


class WanSubgoalGenerator:
    """Generate subgoal images using Wan I2V model."""

    def __init__(
        self,
        checkpoint_dir: str,
        device_id: int = 0,
        offload_model: bool = True
    ):
        """
        Initialize Wan model.

        Args:
            checkpoint_dir: Path to Wan2.2-TI2V-5B checkpoint
            device_id: GPU device ID
            offload_model: Offload model to CPU to save VRAM
        """
        self.checkpoint_dir = checkpoint_dir
        self.device_id = device_id
        self.offload_model = offload_model
        self.model = None

    def load_model(self):
        """Load Wan model (lazy loading)."""
        if self.model is not None:
            return

        from wan.configs.wan_ti2v_5B import wan_ti2v_5B
        from wan.textimage2video import WanTI2V

        print(f"Loading Wan model from {self.checkpoint_dir}...")
        self.model = WanTI2V(
            config=wan_ti2v_5B,
            checkpoint_dir=self.checkpoint_dir,
            device_id=self.device_id,
            t5_cpu=True,
            init_on_cpu=True,
        )
        print("Model loaded.")

    def generate_subgoal_video(
        self,
        first_frame: np.ndarray,
        subgoal_description: str,
        frame_num: int = 17,  # Generate short video (4n+1)
        sampling_steps: int = 30,
        guide_scale: float = 5.0,
        seed: int = 42
    ) -> np.ndarray:
        """
        Generate video showing transition to subgoal state.

        Args:
            first_frame: Initial RGB image (H, W, 3)
            subgoal_description: Text description of target state
            frame_num: Number of frames to generate
            sampling_steps: Diffusion sampling steps
            guide_scale: Classifier-free guidance scale
            seed: Random seed

        Returns:
            Generated video as numpy array (T, H, W, 3)
        """
        self.load_model()

        # Convert to PIL Image
        pil_image = Image.fromarray(first_frame)

        # Build prompt with scene description
        prompt = (
            "The whole scene is in a realistic, industrial art style with three views: "
            "a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
            f"The aloha robot achieves the following state: {subgoal_description}"
        )

        print(f"Generating video for: {subgoal_description[:50]}...")

        # Generate video
        video_tensor = self.model.generate(
            input_prompt=prompt,
            img=pil_image,
            frame_num=frame_num,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            seed=seed,
            offload_model=self.offload_model
        )

        if video_tensor is None:
            raise RuntimeError("Video generation failed")

        # Convert tensor to numpy: (C, T, H, W) -> (T, H, W, C)
        video = video_tensor.permute(1, 2, 3, 0).cpu().numpy()
        video = ((video + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

        return video

    def extract_subgoal_frame(self, video: np.ndarray, frame_idx: int = -1) -> np.ndarray:
        """Extract target frame from generated video (default: last frame)."""
        return video[frame_idx]


def generate_subgoal_images(
    first_frame: np.ndarray,
    subgoals: List[Dict],
    generator: Optional[WanSubgoalGenerator] = None,
    checkpoint_dir: Optional[str] = None,
    output_dir: Optional[str] = None,
    use_wan: bool = False
) -> List[np.ndarray]:
    """
    Generate images for each subgoal.

    Args:
        first_frame: Initial frame (H, W, 3) RGB
        subgoals: List of subgoal dictionaries
        generator: WanSubgoalGenerator instance (optional)
        checkpoint_dir: Path to Wan checkpoint (if generator not provided)
        output_dir: Directory to save outputs
        use_wan: Whether to use Wan model (False = placeholder mode)

    Returns:
        List of subgoal images
    """
    subgoal_images = []

    if use_wan:
        if generator is None:
            if checkpoint_dir is None:
                raise ValueError("checkpoint_dir required when use_wan=True")
            generator = WanSubgoalGenerator(checkpoint_dir)

    for i, subgoal in enumerate(subgoals):
        print(f"\nProcessing subgoal {i+1}/{len(subgoals)}: {subgoal['subgoal_name']}")

        if use_wan and generator is not None:
            # Generate using Wan model
            video = generator.generate_subgoal_video(
                first_frame,
                subgoal['semantic_image_description']
            )
            subgoal_image = generator.extract_subgoal_frame(video)
        else:
            # Placeholder mode - create annotated placeholder
            subgoal_image = first_frame.copy()
            # Add subgoal info as overlay
            h, w = subgoal_image.shape[:2]
            overlay = subgoal_image.copy()
            cv2.rectangle(overlay, (10, h-100), (w-10, h-10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, subgoal_image, 0.5, 0, subgoal_image)

            # Add text
            text = f"Subgoal {subgoal['subgoal_id']}: {subgoal['subgoal_name']}"
            cv2.putText(subgoal_image, text, (20, h-70),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            desc_short = subgoal['semantic_image_description'][:60] + "..."
            cv2.putText(subgoal_image, desc_short, (20, h-40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        subgoal_images.append(subgoal_image)

        # Save if output_dir provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"subgoal_{subgoal['subgoal_id']:02d}_{subgoal['subgoal_name']}.jpg")
            cv2.imwrite(out_path, cv2.cvtColor(subgoal_image, cv2.COLOR_RGB2BGR))
            print(f"  Saved: {out_path}")

    return subgoal_images


def visualize_subgoals(
    first_frame: np.ndarray,
    subgoal_images: List[np.ndarray],
    subgoals: List[Dict],
    output_path: str
):
    """
    Create visualization of first frame + all subgoals.

    Layout: First frame on left, subgoals in a column on right
    """
    n_subgoals = len(subgoal_images)
    h, w = first_frame.shape[:2]

    # Calculate canvas size
    subgoal_h = h // n_subgoals
    canvas_w = w * 2 + 20  # first frame + gap + subgoals
    canvas_h = h + 60  # title space

    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # Add title
    cv2.putText(canvas, "Subgoal Generation: stack_bowls_two", (10, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Add first frame
    canvas[60:60+h, 0:w] = first_frame
    cv2.putText(canvas, "Initial State", (10, 60+h+20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Add subgoals
    x_offset = w + 20
    for i, (img, sg) in enumerate(zip(subgoal_images, subgoals)):
        y_start = 60 + i * subgoal_h
        y_end = y_start + subgoal_h - 5

        # Resize subgoal image to fit
        img_resized = cv2.resize(img, (w, subgoal_h - 5))
        canvas[y_start:y_end, x_offset:x_offset+w] = img_resized

        # Add label
        label = f"{sg['subgoal_id']}. {sg['subgoal_name']}"
        cv2.putText(canvas, label, (x_offset + 5, y_start + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Save
    cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
    print(f"\nVisualization saved to: {output_path}")

    return canvas


def main():
    parser = argparse.ArgumentParser(description="Generate subgoal images using Wan model")

    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--first-frame", type=str, help="Path to first frame image")
    input_group.add_argument("--video", type=str, help="Path to video file")
    input_group.add_argument("--hdf5", type=str, help="Path to HDF5 file")

    # Model options
    parser.add_argument("--checkpoint-dir", type=str,
                       default="/mnt/shared-storage-user/qinyiran/cyujie/cyujie/mounts/qinyiran/vidar/Wan2.2-TI2V-5B",
                       help="Path to Wan2.2-TI2V-5B checkpoint")
    parser.add_argument("--use-wan", action="store_true",
                       help="Use Wan model for generation (requires GPU)")

    # Output options
    parser.add_argument("--output-dir", type=str, default=str(SCRIPT_DIR),
                       help="Output directory")
    parser.add_argument("--subgoals-json", type=str,
                       help="Path to custom subgoals JSON file")

    # Generation options
    parser.add_argument("--sampling-steps", type=int, default=30)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Load or create first frame
    try:
        first_frame = load_first_frame(
            first_frame_path=args.first_frame,
            video_path=args.video,
            hdf5_path=args.hdf5
        )
        print(f"Loaded first frame: {first_frame.shape}")
    except ValueError as e:
        print(f"Warning: {e}")
        print("Creating placeholder image...")
        first_frame = create_placeholder_image((640, 720), "No input image provided")

    # Save first frame
    first_frame_path = os.path.join(args.output_dir, "first_frame.jpg")
    os.makedirs(args.output_dir, exist_ok=True)
    cv2.imwrite(first_frame_path, cv2.cvtColor(first_frame, cv2.COLOR_RGB2BGR))
    print(f"First frame saved to: {first_frame_path}")

    # Load subgoals
    if args.subgoals_json and os.path.exists(args.subgoals_json):
        with open(args.subgoals_json, 'r') as f:
            subgoals = json.load(f)
    else:
        subgoals = DEFAULT_SUBGOALS
        # Save default subgoals
        subgoals_path = os.path.join(args.output_dir, "subgoals.json")
        with open(subgoals_path, 'w') as f:
            json.dump(subgoals, f, indent=2)
        print(f"Subgoals saved to: {subgoals_path}")

    print(f"\nSubgoals to generate: {len(subgoals)}")
    for sg in subgoals:
        print(f"  {sg['subgoal_id']}. {sg['subgoal_name']}")

    # Generate subgoal images
    generator = None
    if args.use_wan:
        generator = WanSubgoalGenerator(
            checkpoint_dir=args.checkpoint_dir,
            device_id=0,
            offload_model=True
        )

    subgoal_images = generate_subgoal_images(
        first_frame=first_frame,
        subgoals=subgoals,
        generator=generator,
        output_dir=args.output_dir,
        use_wan=args.use_wan
    )

    # Create visualization
    vis_path = os.path.join(args.output_dir, "subgoals_visualization.jpg")
    visualize_subgoals(first_frame, subgoal_images, subgoals, vis_path)

    print("\nDone!")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
