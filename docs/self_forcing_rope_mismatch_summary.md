# Self-Forcing Training Pipeline: RoPE Mismatch Summary

## Overview

This document summarizes the key mismatches identified between training and evaluation pipelines in the self-forcing causal diffusion model, specifically focusing on RoPE (Rotary Position Embedding) temporal positional encoding.

---

## 1. Block Index Range Mismatch

### Problem

Training and evaluation use different ranges of `block_idx` values for RoPE position encoding.

| Pipeline | block_idx Values | Latent Frames |
|----------|------------------|---------------|
| Training | None, 1, 2, 3 | 4 (from 16 image frames) |
| Evaluation | None, 1, 2, 3, 4, 5, 6, 7, 8+ | 10+ (from 40+ image frames) |

### Root Cause

- Training config: `num_frames=16` → 4 latent frames (VAE stride = 4)
- Evaluation generates longer sequences (40+ frames → 10+ latent frames)
- Model never sees `block_idx >= 4` during training

### Impact

- RoPE extrapolation to unseen positions during evaluation
- Potential temporal coherence degradation for frames beyond training range
- Attention patterns may not generalize to longer sequences

### Recommendation

Increase training `num_frames` to match evaluation sequence length:
```yaml
data:
  num_frames: 64  # or 81 for 20+ latent frames
```

---

## 2. KV Cache Length Mismatch

### Problem

| Pipeline | Max kv_cache_len |
|----------|------------------|
| Training | 1840 |
| Evaluation | 4140+ |

### Root Cause

- KV cache length = `num_latent_frames × block_size`
- Training: 4 latent frames × 460 tokens/frame = 1840
- Evaluation: 9+ latent frames × 460 tokens/frame = 4140+

### Impact

- Model not trained on attention patterns over longer KV caches
- May affect causal attention behavior for distant frame dependencies

---

## 3. Prefill Mode Inconsistency (Previously Fixed)

### Original Problem

Training initially used `prefill=False` for the first chunk, causing all frames in chunk 0 to receive `block_idx=0` (same RoPE position).

### Fix Applied

Changed to `prefill=True` with causal attention mask for the first chunk:

```python
# wrapper_causal.py - forward_self_forcing
if chunk_idx == 0:
    # First chunk: prefill mode with full sequence
    block_args = {"prefill": True}  # No block_idx, proper per-frame RoPE
else:
    # Subsequent chunks: cache mode with block_idx
    block_args = {"block_size": block_size, "block_idx": chunk_idx}
```

### Evaluation Behavior

- First frame: `block_idx=None` (prefill mode)
- Subsequent frames: `block_idx=1, 2, 3, ...` (cache mode)

This now matches the training pattern.

---

## 4. Chunk Size Configuration

### Training Debug Analysis

With `chunk_size=1` (per-frame processing):
- chunk_idx 0: prefill=True, block_idx=None
- chunk_idx 1: block_idx=1
- chunk_idx 2: block_idx=2
- chunk_idx 3: block_idx=3

With `chunk_size=4` (process 4 frames at once):
- Only 1 chunk total (all 4 latent frames)
- chunk_idx 0: prefill=True, processes all frames together

### Recommendation

For consistency with per-frame autoregressive evaluation, use `chunk_size=1`:
```yaml
self_forcing:
  chunk_size: 1  # Matches eval's per-frame generation
```

---

## 5. Summary of Required Changes

### High Priority

1. **Increase num_frames**: `16 → 64` (or `81`) to expose model to longer block_idx values
2. **Verify chunk_size=1**: Ensures training matches evaluation's per-frame generation pattern

### Already Fixed

- Prefill mode for first chunk (ensures proper RoPE positions)
- Causal attention mask consistency

### Verification Commands

Training debug:
```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 training/train_vidarc.py \
    --config configs/vidarc_2xh200.yaml 2>&1 | grep -E "block_idx|kv_cache"
```

Evaluation debug:
```bash
python vidar/wan/textimage2video_causal_server.py \
    --config configs/eval_config.yaml 2>&1 | grep -E "block_idx|kv_cache"
```

---

## 6. Code References

| File | Line | Function | Purpose |
|------|------|----------|---------|
| `vidar/wan/modules/model_causal.py` | 91-126 | `rope_apply_one` | Core RoPE application with block_idx |
| `vidar-robotwin/training/models/wrapper_causal.py` | 389-442 | `forward_self_forcing` | Training flow with chunk processing |
| `vidar/wan/textimage2video_causal_server.py` | 845 | generation loop | Evaluation block_idx assignment |

---

## 7. Expected Behavior After Fixes

| Aspect | Training | Evaluation | Match |
|--------|----------|------------|-------|
| block_idx range | 0-15 (with num_frames=64) | 0-9 | ✓ |
| First frame RoPE | prefill=True, block_idx=None | prefill, block_idx=None | ✓ |
| Subsequent frames | block_idx=1,2,3,... | block_idx=1,2,3,... | ✓ |
| Attention mode | causal | causal | ✓ |
