"""
Exp 0.6: DINOv2 Embedding Deviation Gate Validation

Validates whether DINOv2 embeddings can distinguish:
  1. SAME phase boundary across different episodes (should be HIGH similarity)
  2. DIFFERENT phase frames within same episode (should be LOW similarity)

If yes → DINOv2 cosine similarity can serve as a lightweight (~5ms)
phase transition gate, replacing the slow VLM (~2s) for online use.

Uses GT demo HDF5 data only (no eval data needed).

Usage:
  python test_deviation_gate.py \
    --data_dir /path/to/hdf5 --task stack_bowls_two \
    --model_path /path/to/dinov2_vits14_pretrain.pth
"""

import os
import sys
import argparse
import time
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

# Reuse gripper boundary detection from build_library
sys.path.insert(0, str(Path(__file__).parent))
from build_library import detect_boundaries_gripper, GRIPPER_INDICES


# ─── DINOv2 ViT-S/14 Architecture (inline, no internet needed) ───

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self, dim, num_heads=6):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if return_attn:
            return x, attn  # attn: (B, num_heads, N, N)
        return x


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attn=True)
            x = x + self.ls1(attn_out)
            x = x + self.ls2(self.mlp(self.norm2(x)))
            return x, attn_weights
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size=14, embed_dim=384):
        super().__init__()
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (B, 3, H, W) -> (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


class DINOv2ViTSmall(nn.Module):
    """DINOv2 ViT-S/14 — self-contained, no torch.hub or internet needed.

    Architecture: 12 blocks, 384 dim, 6 heads, patch_size=14
    Pretrained pos_embed is for 518x518 (37x37=1369 patches).
    Forward pass interpolates pos_embed for any input size.
    """

    def __init__(self, patch_size=14, embed_dim=384, depth=12, num_heads=6):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # Pretrained checkpoint has 1370 positions (1 cls + 37*37 patches for 518px)
        self.pos_embed = nn.Parameter(torch.zeros(1, 1370, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def interpolate_pos_encoding(self, x, h, w):
        """Interpolate pos_embed from pretrained grid to current input grid."""
        num_patches = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1  # 1369 for pretrained

        if num_patches == N and h == w:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]        # (1, 1, D)
        patch_pos = self.pos_embed[:, 1:]      # (1, N, D)

        M = int(N ** 0.5)  # 37 for pretrained
        patch_pos = patch_pos.reshape(1, M, M, self.embed_dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos.float(), size=(h, w),
            mode='bicubic', align_corners=False,
        ).to(patch_pos.dtype)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, self.embed_dim)

        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x, return_patches=False):
        B = x.shape[0]
        # Patch embedding
        x = self.patch_embed(x)         # (B, N, D)
        h = w = int(x.shape[1] ** 0.5)  # grid size

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)

        # Add interpolated position encoding
        x = x + self.interpolate_pos_encoding(x, h, w)

        # Transformer blocks — get last-layer attention if needed
        if return_patches:
            for blk in self.blocks[:-1]:
                x = blk(x)
            x, last_attn = self.blocks[-1](x, return_attn=True)
            # last_attn: (B, num_heads, N+1, N+1)
            x = self.norm(x)
            cls_token = x[:, 0]          # (B, D)
            patch_tokens = x[:, 1:]      # (B, N, D)
            # CLS→patch attention: avg over heads, take CLS row, exclude CLS col
            cls_attn = last_attn[:, :, 0, 1:].mean(dim=1)  # (B, N)
            return cls_token, patch_tokens, cls_attn, h, w
        else:
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x[:, 0]  # CLS token → (B, D)


# ─── Image Encoder ────────────────────────────────────────────────

class ImageEncoder:
    """Lightweight image encoder for deviation gating."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = device

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"DINOv2 weights not found: {model_path}\n"
                f"Download: wget https://dl.fbaipublicfiles.com/dinov2/"
                f"dinov2_vits14/dinov2_vits14_pretrain.pth"
            )

        from torchvision import transforms
        self._transform = transforms.Compose([
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        print(f"  Loading DINOv2 ViT-S/14 from: {model_path}")
        self.model = DINOv2ViTSmall()
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        msg = self.model.load_state_dict(state_dict, strict=False)
        if msg.missing_keys:
            print(f"  Warning: missing keys: {msg.missing_keys}")
        if msg.unexpected_keys:
            # mask_token is expected to be unused at inference
            unexpected = [k for k in msg.unexpected_keys if k != 'mask_token']
            if unexpected:
                print(f"  Warning: unexpected keys: {unexpected}")
        self.model = self.model.to(device)
        self.model.eval()
        self.embed_dim = 384
        print(f"  Loaded DINOv2 on {device}, embed_dim={self.embed_dim}")

    @torch.no_grad()
    def encode(self, images: List[Image.Image]) -> torch.Tensor:
        """Encode images to normalized embeddings. Returns (N, D) tensor."""
        tensors = torch.stack([self._transform(img) for img in images]).to(self.device)
        features = self.model(tensors)
        return F.normalize(features, dim=-1)

    @torch.no_grad()
    def encode_single(self, image: Image.Image) -> torch.Tensor:
        """Encode single image. Returns (D,) tensor."""
        return self.encode([image])[0]

    @torch.no_grad()
    def encode_patches(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """Encode image returning patch tokens + CLS attention map.

        Returns:
            cls_emb:      (D,) normalized CLS embedding
            patch_tokens: (N, D) normalized patch embeddings
            cls_attn:     (N,) CLS→patch attention weights (sums to ~1)
            h, w:         patch grid dimensions
        """
        tensor = self._transform(image).unsqueeze(0).to(self.device)
        cls_emb, patch_tokens, cls_attn, h, w = self.model(tensor, return_patches=True)
        cls_emb = F.normalize(cls_emb, dim=-1)[0]            # (D,)
        patch_tokens = F.normalize(patch_tokens, dim=-1)[0]   # (N, D)
        cls_attn = cls_attn[0]                                 # (N,)
        return cls_emb, patch_tokens, cls_attn, h, w

    def cosine_similarity(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
        """Cosine similarity between two embeddings."""
        return (emb_a @ emb_b).item()


# ─── HDF5 Data Loading ───────────────────────────────────────────

def load_episodes(data_dir: str, task: str, max_episodes: int = 20
                  ) -> List[Dict]:
    """Load images and actions from HDF5 episodes."""
    import h5py

    task_dir = Path(data_dir) / task
    episodes = sorted(task_dir.glob("episode_*.hdf5"))
    if not episodes:
        episodes = sorted(task_dir.glob("episode*.hdf5"))
    if not episodes:
        raise FileNotFoundError(f"No HDF5 files in {task_dir}")

    episodes = episodes[:max_episodes]
    print(f"Loading {len(episodes)} episodes from {task_dir}")

    result = []
    for ep_path in episodes:
        with h5py.File(ep_path, "r") as f:
            # Load actions
            actions = f["action"][:] if "action" in f else f["actions"][:]

            # Load images
            if "observations/unified_image" in f:
                images = f["observations/unified_image"][:]
            elif "observations/images/cam_high" in f:
                images = f["observations/images/cam_high"][:]
            else:
                print(f"  {ep_path.stem}: no image data, skipping")
                continue

            # Detect phase boundaries
            boundaries = detect_boundaries_gripper(actions)

        result.append({
            "name": ep_path.stem,
            "images": images,       # (T, H, W, 3) uint8
            "actions": actions,     # (T, 14 or 16)
            "boundaries": boundaries,
            "num_frames": len(images),
        })
        print(f"  {ep_path.stem}: {len(images)} frames, "
              f"{len(boundaries)-1} phases, boundaries={boundaries}")

    return result


def frame_to_pil(image: np.ndarray) -> Image.Image:
    """Convert HDF5 image array to PIL Image."""
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    # Handle channel-first format
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return Image.fromarray(image, "RGB")


# ─── Test 1: Cross-Episode Same-Phase Similarity ─────────────────

def test_cross_episode_similarity(episodes: List[Dict], encoder: ImageEncoder
                                  ) -> Dict:
    """
    For each phase boundary, compute similarity between the same boundary
    across different episodes.

    Expected: HIGH similarity (same semantic state, different visual details)
    """
    print("\n" + "="*60)
    print("Test 1: Cross-Episode Same-Phase Similarity")
    print("="*60)

    from collections import Counter
    phase_counts = [len(ep["boundaries"]) - 1 for ep in episodes]
    canonical = Counter(phase_counts).most_common(1)[0][0]
    canonical_eps = [ep for ep in episodes if len(ep["boundaries"]) - 1 == canonical]

    print(f"Canonical phases: {canonical}, episodes: {len(canonical_eps)}")

    phase_sims = {}

    for phase_id in range(canonical + 1):
        embeddings = []
        for ep in canonical_eps:
            b_idx = ep["boundaries"][min(phase_id, len(ep["boundaries"])-1)]
            b_idx = min(b_idx, ep["num_frames"] - 1)
            img = frame_to_pil(ep["images"][b_idx])
            emb = encoder.encode_single(img)
            embeddings.append(emb)

        sims = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = encoder.cosine_similarity(embeddings[i], embeddings[j])
                sims.append(sim)

        phase_sims[phase_id] = sims
        if sims:
            print(f"  Boundary {phase_id}: mean={np.mean(sims):.4f}, "
                  f"std={np.std(sims):.4f}, min={np.min(sims):.4f}, "
                  f"max={np.max(sims):.4f} (n={len(sims)} pairs)")

    return phase_sims


# ─── Test 2: Within-Episode Different-Phase Similarity ────────────

def test_different_phase_similarity(episodes: List[Dict], encoder: ImageEncoder
                                    ) -> Dict:
    """
    For each episode, compute similarity between frames at DIFFERENT
    phase boundaries.

    Expected: LOWER similarity than same-phase cross-episode.
    """
    print("\n" + "="*60)
    print("Test 2: Within-Episode Different-Phase Similarity")
    print("="*60)

    all_sims = []

    for ep in episodes[:5]:
        boundary_imgs = []
        for b_idx in ep["boundaries"]:
            b_idx = min(b_idx, ep["num_frames"] - 1)
            img = frame_to_pil(ep["images"][b_idx])
            emb = encoder.encode_single(img)
            boundary_imgs.append(emb)

        for i in range(len(boundary_imgs)):
            for j in range(i + 1, len(boundary_imgs)):
                sim = encoder.cosine_similarity(boundary_imgs[i], boundary_imgs[j])
                all_sims.append({
                    "episode": ep["name"],
                    "boundary_i": i,
                    "boundary_j": j,
                    "distance": abs(j - i),
                    "similarity": sim,
                })

    if all_sims:
        sims_arr = [s["similarity"] for s in all_sims]
        print(f"  All different-phase pairs: mean={np.mean(sims_arr):.4f}, "
              f"std={np.std(sims_arr):.4f}")

        for dist in sorted(set(s["distance"] for s in all_sims)):
            d_sims = [s["similarity"] for s in all_sims if s["distance"] == dist]
            print(f"  Distance {dist}: mean={np.mean(d_sims):.4f}, "
                  f"std={np.std(d_sims):.4f} (n={len(d_sims)})")

    return all_sims


# ─── Test 3: Boundary vs Mid-Phase Similarity ────────────────────

def test_boundary_vs_midphase(episodes: List[Dict], encoder: ImageEncoder
                              ) -> Dict:
    """
    Compare:
      A) Boundary frame of ep X vs boundary frame of ep Y (same phase)
      B) Boundary frame of ep X vs MID-PHASE frame of ep Y (shifted)

    Expected: A > B → similarity peaks at actual transition point.
    """
    print("\n" + "="*60)
    print("Test 3: Boundary vs Mid-Phase Discrimination")
    print("="*60)

    from collections import Counter
    phase_counts = [len(ep["boundaries"]) - 1 for ep in episodes]
    canonical = Counter(phase_counts).most_common(1)[0][0]
    canonical_eps = [ep for ep in episodes if len(ep["boundaries"]) - 1 == canonical]

    if len(canonical_eps) < 2:
        print("  Need at least 2 canonical episodes")
        return {}

    results = {"boundary_match": [], "midphase_match": []}

    for phase_id in range(1, canonical):
        ref_ep = canonical_eps[0]
        ref_b = min(ref_ep["boundaries"][phase_id], ref_ep["num_frames"] - 1)
        ref_emb = encoder.encode_single(frame_to_pil(ref_ep["images"][ref_b]))

        for ep in canonical_eps[1:]:
            # A: Same boundary
            other_b = min(ep["boundaries"][phase_id], ep["num_frames"] - 1)
            boundary_emb = encoder.encode_single(frame_to_pil(ep["images"][other_b]))
            results["boundary_match"].append(
                encoder.cosine_similarity(ref_emb, boundary_emb))

            # B: Mid-phase
            start_b = ep["boundaries"][phase_id - 1]
            end_b = ep["boundaries"][phase_id]
            mid = min((start_b + end_b) // 2, ep["num_frames"] - 1)
            mid_emb = encoder.encode_single(frame_to_pil(ep["images"][mid]))
            results["midphase_match"].append(
                encoder.cosine_similarity(ref_emb, mid_emb))

    bm = results["boundary_match"]
    mm = results["midphase_match"]
    if bm and mm:
        print(f"  Boundary-to-Boundary: mean={np.mean(bm):.4f}, std={np.std(bm):.4f}")
        print(f"  Boundary-to-MidPhase: mean={np.mean(mm):.4f}, std={np.std(mm):.4f}")
        gap = np.mean(bm) - np.mean(mm)
        print(f"  Gap (B2B - B2M):      {gap:+.4f}")
        print(f"  Separable: {'YES' if gap > 0.02 else 'MARGINAL' if gap > 0 else 'NO'}")

    return results


# ─── Test 4: Simulated Gating Accuracy ───────────────────────────

def test_gating_accuracy(episodes: List[Dict], encoder: ImageEncoder) -> Dict:
    """
    Leave-one-out: use episode 0's boundaries as library keyframes,
    check if similarity peaks near actual boundaries in other episodes.
    """
    print("\n" + "="*60)
    print("Test 4: Simulated Gating Accuracy (Leave-One-Out)")
    print("="*60)

    from collections import Counter
    phase_counts = [len(ep["boundaries"]) - 1 for ep in episodes]
    canonical = Counter(phase_counts).most_common(1)[0][0]
    canonical_eps = [ep for ep in episodes if len(ep["boundaries"]) - 1 == canonical]

    if len(canonical_eps) < 2:
        print("  Need at least 2 canonical episodes")
        return {}

    # Library keyframes from episode 0
    ref_ep = canonical_eps[0]
    ref_keyframe_embs = []
    for b_idx in ref_ep["boundaries"]:
        b_idx = min(b_idx, ref_ep["num_frames"] - 1)
        emb = encoder.encode_single(frame_to_pil(ref_ep["images"][b_idx]))
        ref_keyframe_embs.append(emb)

    all_traces = []
    for ep in canonical_eps[1:6]:
        print(f"\n  [{ep['name']}] boundaries={ep['boundaries']}")

        step = max(1, ep["num_frames"] // 40)
        frame_indices = list(range(0, ep["num_frames"], step))

        pil_frames = [frame_to_pil(ep["images"][i]) for i in frame_indices]
        t0 = time.time()
        frame_embs = encoder.encode(pil_frames)
        encode_time = time.time() - t0
        print(f"    Encoded {len(frame_indices)} frames in {encode_time:.2f}s "
              f"({encode_time/len(frame_indices)*1000:.1f}ms/frame)")

        for phase_id in range(1, min(canonical + 1, len(ref_keyframe_embs))):
            kf_emb = ref_keyframe_embs[phase_id]
            sims = (frame_embs @ kf_emb).cpu().numpy()

            actual_boundary = ep["boundaries"][phase_id] if phase_id < len(ep["boundaries"]) else ep["num_frames"] - 1

            peak_frame_local = np.argmax(sims)
            peak_frame_global = frame_indices[peak_frame_local]
            peak_sim = sims[peak_frame_local]
            distance = abs(peak_frame_global - actual_boundary)

            all_traces.append({
                "episode": ep["name"],
                "phase_id": phase_id,
                "actual_boundary": actual_boundary,
                "peak_frame": peak_frame_global,
                "peak_sim": float(peak_sim),
                "distance": distance,
                "total_frames": ep["num_frames"],
            })

            print(f"    Phase {phase_id}: actual={actual_boundary}, "
                  f"peak={peak_frame_global} (sim={peak_sim:.4f}), "
                  f"distance={distance} frames")

    if all_traces:
        distances = [t["distance"] for t in all_traces]
        peak_sims = [t["peak_sim"] for t in all_traces]

        print(f"\n  --- Summary ---")
        print(f"  Peak similarity: mean={np.mean(peak_sims):.4f}, "
              f"std={np.std(peak_sims):.4f}")
        print(f"  Boundary distance: mean={np.mean(distances):.1f}, "
              f"median={np.median(distances):.0f}, max={np.max(distances)} frames")

        for tol in [2, 4, 6, 8]:
            correct = sum(1 for d in distances if d <= tol)
            print(f"  Within ±{tol} frames: {correct}/{len(distances)} "
                  f"({100*correct/len(distances):.0f}%)")

    return {"traces": all_traces}


# ─── Test 5: Latency Benchmark ───────────────────────────────────

def test_latency(encoder: ImageEncoder, image_size: Tuple[int, int] = (720, 640)):
    """Benchmark single-image encoding latency."""
    print("\n" + "="*60)
    print("Test 5: Latency Benchmark")
    print("="*60)

    dummy = Image.fromarray(np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8))

    # Warmup
    for _ in range(3):
        encoder.encode_single(dummy)

    latencies = []
    for _ in range(20):
        t0 = time.time()
        encoder.encode_single(dummy)
        latencies.append((time.time() - t0) * 1000)

    print(f"  Image size: {image_size}")
    print(f"  Mean: {np.mean(latencies):.1f}ms")
    print(f"  Std:  {np.std(latencies):.1f}ms")
    print(f"  Min:  {np.min(latencies):.1f}ms")
    print(f"  Max:  {np.max(latencies):.1f}ms")
    print(f"  P95:  {np.percentile(latencies, 95):.1f}ms")

    feasible = np.mean(latencies) < 50
    print(f"  Online feasible (<50ms): {'YES' if feasible else 'NO'}")

    return {"mean_ms": float(np.mean(latencies)), "latencies": latencies}


# ─── Video Loading ────────────────────────────────────────────────

def load_video_frames(video_path: str) -> List[np.ndarray]:
    """Load all frames from mp4 as numpy arrays (RGB)."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def find_pred_video(eval_dir: str, ep_id: int) -> Optional[str]:
    """Find pred video for an episode, trying multiple naming patterns.

    Tries in order:
      1. episode{N}_pred_160.mp4  (160 frames, standard)
      2. episode{N}_mask_64.mp4   (64 frames, alternative)
    Returns the first existing path, or None.
    """
    candidates = [
        os.path.join(eval_dir, f"episode{ep_id}_pred_160.mp4"),
        os.path.join(eval_dir, f"episode{ep_id}_mask_64.mp4"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def align_frame_indices(n_pred: int, n_actual: int, num_samples: int) -> List[Tuple[int, int]]:
    """Generate aligned (pred_idx, actual_idx) pairs for proportional sampling.

    When pred and actual have different frame counts (e.g. 64 vs 160),
    sample proportionally through both videos rather than truncating.

    Returns list of (pred_frame_idx, actual_frame_idx) tuples.
    """
    sample_fracs = np.linspace(0, 1, num_samples)
    pairs = []
    for frac in sample_fracs:
        p_idx = min(int(frac * (n_pred - 1)), n_pred - 1)
        a_idx = min(int(frac * (n_actual - 1)), n_actual - 1)
        pairs.append((p_idx, a_idx))
    return pairs


def apply_embodiment_mask(image: np.ndarray, mask_ratio: float = 0.4) -> np.ndarray:
    """Mask top portion of image (robot arms region)."""
    masked = image.copy()
    h = masked.shape[0]
    masked[:int(h * mask_ratio), :] = 0
    return masked


# ─── Test 6: Predicted vs Actual Frame Comparison ────────────────

def _compute_episode_sims(
    plan_frames: List[np.ndarray],
    actual_frames: List[np.ndarray],
    encoder: ImageEncoder,
    num_samples: int,
    mask_ratio: float,
) -> List[Dict]:
    """Compute per-frame similarities for one episode at one mask ratio."""
    if len(plan_frames) < 2 or len(actual_frames) < 2:
        return []
    pairs = align_frame_indices(len(plan_frames), len(actual_frames), num_samples)
    sims = []
    for p_idx, a_idx in pairs:
        plan_img = plan_frames[p_idx]
        actual_img = actual_frames[a_idx]
        # Crop pred frame to match actual height (pred may have extra arm views)
        if plan_img.shape[0] > actual_img.shape[0]:
            plan_img = plan_img[:actual_img.shape[0], :, :]
        if mask_ratio > 0:
            plan_img = apply_embodiment_mask(plan_img, mask_ratio)
            actual_img = apply_embodiment_mask(actual_img, mask_ratio)
        plan_pil = Image.fromarray(plan_img, "RGB")
        actual_pil = Image.fromarray(actual_img, "RGB")
        plan_emb = encoder.encode_single(plan_pil)
        actual_emb = encoder.encode_single(actual_pil)
        sim = encoder.cosine_similarity(plan_emb, actual_emb)
        sims.append({"frame": int(a_idx), "similarity": sim})
    return sims


def test_predicted_vs_actual(
    eval_dir: str,
    episode_ids: List[int],
    encoder: ImageEncoder,
    num_samples: int = 16,
    mask_ratios: List[float] = [0.0, 0.3, 0.5],
) -> Dict:
    """
    Compare DINOv2 embeddings of predicted frames vs actual frames
    across multiple episodes. Tests with different embodiment mask ratios.

    Expected: high similarity when execution matches plan,
    low similarity when execution diverges (failure).
    """
    print("\n" + "="*60)
    print("Test 6: Predicted vs Actual (Multi-Episode, DINOv2)")
    print("="*60)
    print(f"  Eval dir: {eval_dir}")
    print(f"  Episodes: {episode_ids}")

    # Load all episode video pairs
    episode_data = []
    for ep_id in episode_ids:
        plan_path = find_pred_video(eval_dir, ep_id)
        actual_path = os.path.join(eval_dir, f"episode{ep_id}.mp4")
        if plan_path is None:
            print(f"  WARNING: no pred video for episode {ep_id}, skipping")
            continue
        if not os.path.exists(actual_path):
            print(f"  WARNING: {actual_path} not found, skipping")
            continue
        plan_frames = load_video_frames(plan_path)
        actual_frames = load_video_frames(actual_path)
        ph, pw = plan_frames[0].shape[:2] if plan_frames else (0, 0)
        ah, aw = actual_frames[0].shape[:2] if actual_frames else (0, 0)
        crop_note = f" -> crop pred to {ah}px" if ph > ah else ""
        pred_name = os.path.basename(plan_path)
        print(f"  Episode {ep_id}: pred={pred_name} {len(plan_frames)}f({pw}x{ph}), "
              f"actual={len(actual_frames)}f({aw}x{ah}){crop_note}")
        episode_data.append({
            "id": ep_id,
            "plan_frames": plan_frames,
            "actual_frames": actual_frames,
        })

    if not episode_data:
        print("  No valid episode pairs found")
        return {}

    print(f"  Loaded {len(episode_data)} episode pairs")

    # Results organized by mask ratio
    results_by_mask = {}

    for mask_ratio in mask_ratios:
        label = f"mask={mask_ratio:.0%}" if mask_ratio > 0 else "no_mask"
        print(f"\n  {'='*50}")
        print(f"  {label}")
        print(f"  {'='*50}")

        all_sims = []       # all per-frame sims across episodes
        ep_results = []     # per-episode summaries

        for ep in episode_data:
            sims = _compute_episode_sims(
                ep["plan_frames"], ep["actual_frames"],
                encoder, num_samples, mask_ratio,
            )
            if not sims:
                continue

            sim_values = [s["similarity"] for s in sims]
            ep_mean = np.mean(sim_values)
            ep_std = np.std(sim_values)

            print(f"\n  Episode {ep['id']}:")
            for s in sims:
                bar = "#" * int(s["similarity"] * 40)
                print(f"    frame {s['frame']:>4d}: {s['similarity']:.4f} |{bar}")
            print(f"    mean={ep_mean:.4f}, std={ep_std:.4f}, "
                  f"min={np.min(sim_values):.4f}, max={np.max(sim_values):.4f}")

            # Per-episode anomaly detection
            threshold = ep_mean - 1.5 * ep_std
            anomalies = [s for s in sims if s["similarity"] < threshold]
            if anomalies:
                print(f"    Anomalies (sim < {threshold:.4f}):")
                for a in anomalies:
                    print(f"      frame {a['frame']}: {a['similarity']:.4f}")

            ep_results.append({
                "episode": ep["id"],
                "mean": float(ep_mean),
                "std": float(ep_std),
                "min": float(np.min(sim_values)),
                "max": float(np.max(sim_values)),
                "anomalies": len(anomalies),
                "sims": sims,
            })
            all_sims.extend(sim_values)

        # Cross-episode aggregate
        if all_sims:
            print(f"\n  --- {label} Aggregate ({len(ep_results)} episodes) ---")
            print(f"  Global mean: {np.mean(all_sims):.4f}")
            print(f"  Global std:  {np.std(all_sims):.4f}")
            print(f"  Global min:  {np.min(all_sims):.4f}")
            print(f"  Global max:  {np.max(all_sims):.4f}")

            # Per-episode comparison
            ep_means = [e["mean"] for e in ep_results]
            ep_stds = [e["std"] for e in ep_results]
            print(f"  Per-episode means: {['%.4f' % m for m in ep_means]}")
            print(f"  Per-episode stds:  {['%.4f' % s for s in ep_stds]}")
            print(f"  Cross-episode mean variance: {np.std(ep_means):.4f}")

        results_by_mask[label] = {
            "episodes": ep_results,
            "global_mean": float(np.mean(all_sims)) if all_sims else 0,
            "global_std": float(np.std(all_sims)) if all_sims else 0,
        }

    # Compare mask effect across all episodes
    print(f"\n  {'='*50}")
    print(f"  Mask Comparison (all episodes)")
    print(f"  {'='*50}")
    for label, data in results_by_mask.items():
        total_anomalies = sum(e["anomalies"] for e in data["episodes"])
        print(f"  {label:>10s}: global_mean={data['global_mean']:.4f}, "
              f"global_std={data['global_std']:.4f}, "
              f"total_anomalies={total_anomalies}")

    if len(results_by_mask) >= 2:
        keys = list(results_by_mask.keys())
        std_no_mask = results_by_mask[keys[0]]["global_std"]
        std_best_mask = max(results_by_mask[k]["global_std"] for k in keys[1:])
        if std_best_mask > std_no_mask:
            print(f"  Masking INCREASES variance ({std_no_mask:.4f} -> {std_best_mask:.4f})"
                  f" -> better discrimination")
        else:
            print(f"  Masking does NOT improve variance")

    return results_by_mask


# ─── Test 7: Attention-Weighted Patch Similarity ─────────────────

def _compute_episode_patch_sims(
    plan_frames: List[np.ndarray],
    actual_frames: List[np.ndarray],
    encoder: ImageEncoder,
    num_samples: int,
) -> List[Dict]:
    """Compute attention-weighted patch similarity for one episode.

    For each sampled frame pair:
    1. Extract patch tokens (16x16 grid, 384-dim each) + CLS attention map
    2. Compute per-patch cosine similarity between pred and actual
    3. Weight by CLS attention → focus on object regions
    4. Also report: global CLS sim, unweighted patch mean, min-patch sim
    """
    if len(plan_frames) < 2 or len(actual_frames) < 2:
        return []
    pairs = align_frame_indices(len(plan_frames), len(actual_frames), num_samples)
    results = []
    for p_idx, a_idx in pairs:
        plan_img = plan_frames[p_idx]
        actual_img = actual_frames[a_idx]
        # Crop pred frame to match actual height
        if plan_img.shape[0] > actual_img.shape[0]:
            plan_img = plan_img[:actual_img.shape[0], :, :]

        plan_pil = Image.fromarray(plan_img, "RGB")
        actual_pil = Image.fromarray(actual_img, "RGB")

        # Get patch-level features
        p_cls, p_patches, p_attn, h, w = encoder.encode_patches(plan_pil)
        a_cls, a_patches, a_attn, _, _ = encoder.encode_patches(actual_pil)

        # Per-patch cosine similarity: (N,)
        patch_sim = (p_patches * a_patches).sum(dim=-1)  # already normalized

        # Attention weights: average pred and actual CLS attention
        attn_weights = (p_attn + a_attn) / 2.0
        attn_weights = attn_weights / attn_weights.sum()  # renormalize

        # Weighted similarity
        weighted_sim = (attn_weights * patch_sim).sum().item()

        # Unweighted patch mean
        unweighted_sim = patch_sim.mean().item()

        # CLS global similarity (same as Test 6)
        cls_sim = (p_cls @ a_cls).item()

        # Spatial map for analysis
        patch_sim_map = patch_sim.cpu().numpy().reshape(h, w)
        attn_map = attn_weights.cpu().numpy().reshape(h, w)

        # Top-K attended patches: find patches with highest attention
        topk = min(10, patch_sim.shape[0])
        topk_idx = attn_weights.topk(topk).indices
        topk_sim = patch_sim[topk_idx].mean().item()

        # Bottom quartile of patch similarities (worst matching regions)
        sorted_sims = patch_sim.sort().values
        bottom_q = sorted_sims[:len(sorted_sims) // 4].mean().item()

        results.append({
            "frame": int(a_idx),
            "cls_sim": cls_sim,
            "unweighted_patch_sim": unweighted_sim,
            "weighted_patch_sim": weighted_sim,
            "topk_attended_sim": topk_sim,
            "bottom_quartile_sim": bottom_q,
            "patch_min": patch_sim.min().item(),
            "patch_max": patch_sim.max().item(),
            "patch_grid": (h, w),
        })
    return results


def test_patch_similarity(
    eval_dir: str,
    episode_ids: List[int],
    encoder: ImageEncoder,
    num_samples: int = 16,
) -> Dict:
    """Test 7: Attention-weighted patch-level pred vs actual comparison.

    Instead of comparing global CLS embeddings (Test 6), compares
    patch-level features weighted by CLS attention to focus on objects.
    """
    print("\n" + "=" * 60)
    print("Test 7: Attention-Weighted Patch Similarity (DINOv2)")
    print("=" * 60)
    print(f"  Eval dir: {eval_dir}")
    print(f"  Episodes: {episode_ids}")

    # Load all episode video pairs
    episode_data = []
    for ep_id in episode_ids:
        plan_path = find_pred_video(eval_dir, ep_id)
        actual_path = os.path.join(eval_dir, f"episode{ep_id}.mp4")
        if plan_path is None:
            print(f"  WARNING: no pred video for episode {ep_id}, skipping")
            continue
        if not os.path.exists(actual_path):
            print(f"  WARNING: {actual_path} not found, skipping")
            continue
        plan_frames = load_video_frames(plan_path)
        actual_frames = load_video_frames(actual_path)
        ph, pw = plan_frames[0].shape[:2] if plan_frames else (0, 0)
        ah, aw = actual_frames[0].shape[:2] if actual_frames else (0, 0)
        crop_note = f" -> crop {ah}px" if ph > ah else ""
        pred_name = os.path.basename(plan_path)
        print(f"  Episode {ep_id}: pred={pred_name} {len(plan_frames)}f({pw}x{ph}), "
              f"actual={len(actual_frames)}f({aw}x{ah}){crop_note}")
        episode_data.append({
            "id": ep_id,
            "plan_frames": plan_frames,
            "actual_frames": actual_frames,
        })

    if not episode_data:
        print("  No valid episode pairs found")
        return {}

    print(f"  Loaded {len(episode_data)} episode pairs\n")

    all_ep_results = []

    for ep in episode_data:
        frame_results = _compute_episode_patch_sims(
            ep["plan_frames"], ep["actual_frames"],
            encoder, num_samples,
        )
        if not frame_results:
            continue

        cls_sims = [r["cls_sim"] for r in frame_results]
        weighted_sims = [r["weighted_patch_sim"] for r in frame_results]
        topk_sims = [r["topk_attended_sim"] for r in frame_results]
        bottom_sims = [r["bottom_quartile_sim"] for r in frame_results]

        print(f"  Episode {ep['id']}:")
        print(f"    {'frame':>6s}  {'CLS':>7s}  {'w_patch':>7s}  {'topK':>7s}  {'bot25%':>7s}  {'p_min':>7s}")
        for r in frame_results:
            print(f"    {r['frame']:>6d}  {r['cls_sim']:>7.4f}  "
                  f"{r['weighted_patch_sim']:>7.4f}  "
                  f"{r['topk_attended_sim']:>7.4f}  "
                  f"{r['bottom_quartile_sim']:>7.4f}  "
                  f"{r['patch_min']:>7.4f}")

        ep_summary = {
            "episode": ep["id"],
            "cls_mean": float(np.mean(cls_sims)),
            "weighted_mean": float(np.mean(weighted_sims)),
            "topk_mean": float(np.mean(topk_sims)),
            "bottom_q_mean": float(np.mean(bottom_sims)),
            "weighted_std": float(np.std(weighted_sims)),
            "frames": frame_results,
        }
        print(f"    --- mean:  CLS={ep_summary['cls_mean']:.4f}  "
              f"w_patch={ep_summary['weighted_mean']:.4f}  "
              f"topK={ep_summary['topk_mean']:.4f}  "
              f"bot25%={ep_summary['bottom_q_mean']:.4f}")
        all_ep_results.append(ep_summary)

    # Cross-episode summary
    if all_ep_results:
        print(f"\n  {'=' * 50}")
        print(f"  Cross-Episode Summary ({len(all_ep_results)} episodes)")
        print(f"  {'=' * 50}")
        print(f"  {'ep':>4s}  {'CLS':>7s}  {'w_patch':>7s}  {'topK':>7s}  {'bot25%':>7s}  {'w_std':>7s}")
        for ep in all_ep_results:
            print(f"  ep{ep['episode']:<3d} {ep['cls_mean']:>7.4f}  "
                  f"{ep['weighted_mean']:>7.4f}  "
                  f"{ep['topk_mean']:>7.4f}  "
                  f"{ep['bottom_q_mean']:>7.4f}  "
                  f"{ep['weighted_std']:>7.4f}")

        # Global aggregates
        all_w = [ep["weighted_mean"] for ep in all_ep_results]
        all_topk = [ep["topk_mean"] for ep in all_ep_results]
        all_bot = [ep["bottom_q_mean"] for ep in all_ep_results]
        print(f"\n  Global weighted_patch: mean={np.mean(all_w):.4f}, "
              f"std={np.std(all_w):.4f}, range=[{np.min(all_w):.4f}, {np.max(all_w):.4f}]")
        print(f"  Global topK_attended:  mean={np.mean(all_topk):.4f}, "
              f"std={np.std(all_topk):.4f}, range=[{np.min(all_topk):.4f}, {np.max(all_topk):.4f}]")
        print(f"  Global bottom_25%:     mean={np.mean(all_bot):.4f}, "
              f"std={np.std(all_bot):.4f}, range=[{np.min(all_bot):.4f}, {np.max(all_bot):.4f}]")

    return {"episodes": all_ep_results}


# ─── Main ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Exp 0.6: DINOv2 Deviation Gate Validation")
    parser.add_argument("--data_dir", type=str,
                        default="/mnt/shared-storage-user/kangli/workspace/cyujie/mounts/qinyiran/datasets/robotwin/processed",
                        help="HDF5 data directory")
    parser.add_argument("--task", type=str, default="stack_bowls_two",
                        help="Task name")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to dinov2_vits14_pretrain.pth")
    parser.add_argument("--max_episodes", type=int, default=10,
                        help="Max episodes to load")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument("--tests", nargs="+", type=int, default=[1, 2, 3, 4, 5],
                        help="Which tests to run (1-7)")
    # Test 6: predicted vs actual video comparison (multi-episode)
    parser.add_argument("--eval_dir", type=str, default=None,
                        help="Eval result dir containing episode{N}_pred_160.mp4 and episode{N}.mp4 (Test 6)")
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                        help="Episode IDs to test (Test 6)")
    parser.add_argument("--mask_ratios", nargs="+", type=float,
                        default=[0.0, 0.3, 0.5],
                        help="Embodiment mask ratios to test (Test 6)")
    args = parser.parse_args()

    print(f"DINOv2 Deviation Gate Validation")
    print(f"  Weights: {args.model_path}")
    print(f"  Device: {args.device}")
    print()

    # Load encoder
    print("Loading encoder...")
    t0 = time.time()
    encoder = ImageEncoder(model_path=args.model_path, device=args.device)
    print(f"  Load time: {time.time()-t0:.1f}s\n")

    results = {}

    # Tests 1-5: HDF5 cross-episode tests
    episodes = None
    if any(t in args.tests for t in [1, 2, 3, 4, 5]):
        episodes = load_episodes(args.data_dir, args.task, args.max_episodes)
        if not episodes:
            print("No episodes loaded for tests 1-5.")

    if episodes:
        if 1 in args.tests:
            results["cross_episode"] = test_cross_episode_similarity(episodes, encoder)

        if 2 in args.tests:
            results["different_phase"] = test_different_phase_similarity(episodes, encoder)

        if 3 in args.tests:
            results["boundary_vs_mid"] = test_boundary_vs_midphase(episodes, encoder)

        if 4 in args.tests:
            results["gating_accuracy"] = test_gating_accuracy(episodes, encoder)

        if 5 in args.tests:
            if episodes[0]["images"].shape[1:3]:
                h, w = episodes[0]["images"].shape[1:3]
            else:
                h, w = 720, 640
            results["latency"] = test_latency(encoder, image_size=(h, w))

    # Test 6: predicted vs actual video (multi-episode)
    if 6 in args.tests:
        if not args.eval_dir:
            print("\nTest 6 requires --eval_dir")
        else:
            results["pred_vs_actual"] = test_predicted_vs_actual(
                args.eval_dir, args.episodes, encoder,
                mask_ratios=args.mask_ratios,
            )

    # Test 7: attention-weighted patch similarity
    if 7 in args.tests:
        if not args.eval_dir:
            print("\nTest 7 requires --eval_dir")
        else:
            results["patch_sim"] = test_patch_similarity(
                args.eval_dir, args.episodes, encoder,
            )

    # ─── Final Summary ────────────────────────────────────────────
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)

    if "cross_episode" in results and "different_phase" in results:
        cross_sims = []
        for phase_id, sims in results["cross_episode"].items():
            if phase_id > 0:
                cross_sims.extend(sims)

        diff_sims = [s["similarity"] for s in results["different_phase"]]

        if cross_sims and diff_sims:
            print(f"\n  Same-phase cross-episode:  mean={np.mean(cross_sims):.4f}")
            print(f"  Different-phase same-ep:   mean={np.mean(diff_sims):.4f}")
            margin = np.mean(cross_sims) - np.mean(diff_sims)
            print(f"  Discrimination margin:     {margin:+.4f}")

            if margin > 0.05:
                print(f"  Verdict: STRONG discrimination -> deviation gate feasible")
            elif margin > 0.02:
                print(f"  Verdict: MODERATE discrimination -> may work with careful threshold")
            else:
                print(f"  Verdict: WEAK discrimination -> deviation gate unreliable")

    if "gating_accuracy" in results and results["gating_accuracy"].get("traces"):
        traces = results["gating_accuracy"]["traces"]
        within_4 = sum(1 for t in traces if t["distance"] <= 4)
        print(f"\n  Gating accuracy (±4 frames): {within_4}/{len(traces)} "
              f"({100*within_4/len(traces):.0f}%)")

    if "latency" in results:
        print(f"\n  Encoding latency: {results['latency']['mean_ms']:.1f}ms/frame")

    if "pred_vs_actual" in results:
        print(f"\n  Predicted vs Actual (Test 6, multi-episode):")
        for label, data in results["pred_vs_actual"].items():
            total_anomalies = sum(e["anomalies"] for e in data["episodes"])
            print(f"    {label}: global_mean={data['global_mean']:.4f}, "
                  f"global_std={data['global_std']:.4f}, "
                  f"anomalies={total_anomalies}/{len(data['episodes'])}ep")

    if "patch_sim" in results and results["patch_sim"].get("episodes"):
        eps = results["patch_sim"]["episodes"]
        all_w = [e["weighted_mean"] for e in eps]
        all_topk = [e["topk_mean"] for e in eps]
        print(f"\n  Patch Similarity (Test 7, {len(eps)} episodes):")
        print(f"    weighted_patch: mean={np.mean(all_w):.4f}, std={np.std(all_w):.4f}")
        print(f"    topK_attended:  mean={np.mean(all_topk):.4f}, std={np.std(all_topk):.4f}")

    print()


if __name__ == "__main__":
    main()
