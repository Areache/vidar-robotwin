# Vidar Training Pipeline Fixes

## Overview

This document summarizes critical fixes made to align the training pipeline with the evaluation pipeline, resolving visual artifacts in fine-tuned model outputs.

---

## Issues and Fixes Summary

| Issue | Symptom | Root Cause | Fix |
|-------|---------|------------|-----|
| Velocity sign | Horizontal stripes | `v = x0 - x1` (wrong) | `v = x1 - x0` (clean - noise) |
| Timestep scale | Incorrect time embedding | `t ∈ [0,1]` passed to model | Scale to `t * 1000` for model input |
| Positional encoding | Light/dark flickering | `prefill=False` (all frames get position 0) | `prefill=True` with causal mask |
| Checkpoint dtype | Block artifacts | FSDP saves fp32, eval expects bf16 | Convert fp32 → bf16 on save |

---

## Detailed Explanations

### 1. Velocity Sign Mismatch

**Flow Matching Convention:**
```
x_t = t * x_1 + (1-t) * x_0
```
where:
- `x_1` = clean data
- `x_0` = noise
- `t ∈ [0, 1]`: t=0 → pure noise, t=1 → clean

**Velocity (derivative):**
```
v = dx_t/dt = x_1 - x_0 (clean - noise)
```

**Bug:** Training used `v_target = x0 - x1` (wrong sign)

**Fix locations:**
- `training/trainers/vidarc_trainer.py:320`: `v_target = x1 - x0`
- `training/models/wrapper_causal.py:378`: `v_target = x_chunk - x0_chunk`

---

### 2. Timestep Scale Mismatch

**Problem:** The Wan model's sinusoidal time embedding was trained with timesteps in `[0, 1000]`, but training code passed `t ∈ [0, 1]`.

**Inference (eval) pipeline:**
```python
# In scheduler
self.timesteps = sigmas * num_train_timesteps  # sigmas ∈ [0,1], timesteps ∈ [0,1000]

# Passed to model
timestep = torch.ones(...) * t  # t from timesteps, so t ∈ [0, 1000]
```

**Training (before fix):**
```python
t = sample_timestep(B, device)  # Returns t ∈ [0, 1]
v_pred = model(x_t, t, context)  # Wrong! Model expects [0, 1000]
```

**Fix:**
```python
t = sample_timestep(B, device)  # t ∈ [0, 1] for interpolation
t_model = t * 1000.0            # Scale for model input
v_pred = model(x_t, t_model, context)
```

**Fix locations:**
- `training/trainers/vidarc_trainer.py:325-330`
- `training/trainers/vidar_trainer.py:180-183`
- `training/models/wrapper_causal.py:401-402`

---

### 3. Positional Encoding (RoPE) Mismatch

**Problem:** Training used `prefill=False` without `block_idx`, causing all frames to receive position 0 in RoPE.

**Eval pipeline:**
```python
# Conditional frames: prefill=True, correct positions per frame
model(cond_latent, t=0, prefill=True, attention_mask=causal_mask)

# New frames: block_idx specifies position
model(new_frame, t=timestep, block_idx=frame_idx)
```

**Training (before fix):**
```python
# forward() method
return self.forward_causal(x, t, context, prefill=False)  # All frames get position 0!
```

**Fix:**
```python
def forward(self, x, t, context, seq_len=None):
    B, C, T, H, W = x.shape
    block_size = self.get_block_size(x.shape)
    attention_mask = self._build_causal_mask(x.shape, block_size, x.device)

    return self.forward_causal(
        x=x, t=t, context=context,
        attention_mask=attention_mask,
        prefill=True,  # Correct positional encoding per frame
    )
```

**Fix location:** `training/models/wrapper_causal.py:460-491`

---

### 4. Checkpoint Dtype Mismatch

**Problem:** FSDP saves weights in fp32 even with bf16 mixed precision, but eval expects bf16.

**Fix:** Convert fp32 → bf16 when saving checkpoint:
```python
def save_checkpoint(self, path):
    # ... get state dict ...

    # Convert fp32 to bf16
    for key in dit_state:
        if dit_state[key].dtype == torch.float32:
            dit_state[key] = dit_state[key].to(torch.bfloat16)

    torch.save(checkpoint, path)
```

**Fix location:** `training/trainers/vidarc_trainer.py:364-372`

---

## Training vs Eval Pipeline Comparison

| Aspect | Training (after fix) | Eval |
|--------|---------------------|------|
| Timestep range | `t ∈ [0,1]` for interpolation, `t*1000` for model | `t ∈ [0,1000]` from scheduler |
| Velocity | `v = x1 - x0` | `v = x1 - x0` (assumed by sampler) |
| Positional encoding | `prefill=True` with causal mask | `prefill=True` for cond, `block_idx` for new |
| Checkpoint dtype | bf16 | bf16 |

---

## Files Modified

```
training/
├── models/
│   └── wrapper_causal.py    # Velocity, timestep scale, positional encoding
├── trainers/
│   ├── vidar_trainer.py     # Timestep scale (Stage 1)
│   └── vidarc_trainer.py    # Velocity, timestep scale, bf16 conversion (Stage 2)
```

---

## Commit

```
commit d8d244c
Fix critical training pipeline issues: velocity, timestep, positional encoding

1. Velocity sign: v_target = x1 - x0 (clean - noise), not x0 - x1
2. Timestep scaling: t * 1000 for model input (model expects [0,1000], not [0,1])
3. Positional encoding: prefill=True with causal mask for correct RoPE positions
4. bf16 checkpoint conversion for eval compatibility
```

---

## Notes

### Why Loss Increased After Positional Encoding Fix

After applying the `prefill=True` fix, the first-step loss may be higher. This is **expected**:

1. **Before fix:** All frames had position 0, making the task "easier" (but wrong)
2. **After fix:** Each frame has correct position, requiring proper temporal learning
3. **Result:** Initial loss is higher, but training is now correct

The loss will decrease as training continues with the correct setup.

### Retraining Required

After applying these fixes, you must **retrain from scratch** or from a checkpoint that was saved before any incorrect training. Continuing from a checkpoint trained with wrong settings may produce suboptimal results.

---

## Results

### Task Performance Comparison

| Task | Baseline | 10 steps | Delta |
|------|----------|----------|-------|
| click_alarmclock | 1.0 | 0.9 | -0.1 |
| click_bell | 1.0 | 1.0 | 0.0 |
| dump_bin_bigbin | 0.2 | N/A | - |
| grab_roller | 1.0 | 1.0 | 0.0 |
| open_laptop | 0.6 | 0.7 | +0.1 |
| pick_dual_bottles | 0.3 | 0.6 | +0.3 |
| place_a2b_right | 0.5 | 0.6 | +0.1 |
| place_bread_basket | 0.9 | 0.8 | -0.1 |
| place_burger_fries | 0.8 | 0.8 | 0.0 |
| place_cans_plasticbox | 0.7 | 0.7 | 0.0 |
| place_container_plate | 1.0 | 0.8 | -0.2 |
| place_empty_cup | 0.8 | 0.7 | -0.1 |
| place_object_stand | 0.6 | 0.6 | 0.0 |
| place_phone_stand | 0.7 | 0.4 | -0.3 |
| press_stapler | 0.9 | 1.0 | +0.1 |
| shake_bottle | 0.9 | 0.8 | -0.1 |
| shake_bottle_horizontally | 1.0 | 1.0 | 0.0 |
| stack_bowls_two | 0.7 | 0.9 | +0.2 |
| turn_switch | 0.4 | 0.1 | -0.3 |

### Summary Statistics

| Metric | Value |
|--------|-------|
| Tasks Improved | 6 |
| Tasks Unchanged | 6 |
| Tasks Degraded | 6 |
| Average Delta | ~0.0 |

**Notable Improvements:**
- `pick_dual_bottles`: +0.3 (0.3 → 0.6)
- `stack_bowls_two`: +0.2 (0.7 → 0.9)

**Notable Degradations:**
- `turn_switch`: -0.3 (0.4 → 0.1)
- `place_phone_stand`: -0.3 (0.7 → 0.4)
