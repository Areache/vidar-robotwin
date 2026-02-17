# Training-Eval Alignment Plan (Reconstructed)

## Implementation Status: COMPLETED

### Files Edited

| File | Changes |
|------|---------|
| `vidar-robotwin/training/models/wrapper_causal.py` | Added `forward_self_forcing_aligned()` and `forward_self_forcing_multistep_aligned()` functions with cache pop simulation |
| `vidar/wan/modules/model_causal.py` | Added debug prints to `rope_apply_chunk()`, `chunk_prefill()`, and `Attention.forward()` |
| `vidar/wan/modules/block_attention.py` | Updated `get_flex_block_mask_chunk_prefill()` with better docs and debug prints |

### How to Enable Debug Prints

```bash
# Enable alignment debug prints
export ALIGNMENT_DEBUG=1

# Run training with aligned function
python train.py --use_aligned_self_forcing
```

### New Functions Added

1. **`forward_self_forcing_aligned()`** - Single-step self-forcing with cache pop simulation
2. **`forward_self_forcing_multistep_aligned()`** - Multi-step denoising with cache pop simulation
3. **`_simulate_cache_pop()`** - Helper to simulate eval's cache pop behavior

---

## Executive Summary

Training and evaluation have **5 critical mismatches** that cause distribution shift. This plan addresses each mismatch with specific code changes.

```
┌─────────────────────────────────────────────────────────────────────┐
│  CURRENT STATE                                                       │
│                                                                      │
│  Training: Full causal, chunk_prefill=False, absolute RoPE           │
│  Eval:     Sink+window, chunk_prefill=True, relative RoPE            │
│                                                                      │
│  → Model trained on pattern A, evaluated on pattern B                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Mismatch Overview

| # | Mismatch | Training (Current) | Eval (Actual) | Priority |
|---|----------|-------------------|---------------|----------|
| 1 | `chunk_prefill` flag | Always `False` | `True` after cache pop | **P0** |
| 2 | KV Cache behavior | Grow forever | Pop + rebuild (keep sink) | **P0** |
| 3 | RoPE positioning | Absolute `block_idx` | Relative `kv_num_block` | **P0** |
| 4 | Attention mask | Simple causal | Sink + causal within new | **P1** |
| 5 | Max visible context | All frames (unbounded) | 5-8 frames (bounded) | **P1** |

---

## Mismatch #1: `chunk_prefill` Flag

### Problem

```python
# Training (wrapper_causal.py:394, 402)
chunk_prefill = False  # ALWAYS False, even for subsequent chunks

# Eval (textimage2video_causal_server.py:554)
chunk_prefill = True   # True after cache pop in Round 2+
```

When `chunk_prefill=True`, the model uses a completely different code path:
- Different RoPE function (`rope_apply_chunk` instead of `rope_apply_one`)
- Different attention computation (`chunk_prefill()` method)
- Different mask pattern

### Fix

```python
# In wrapper_causal.py forward_self_forcing()

def forward_self_forcing(self, ...):
    for chunk_idx in range(num_chunks):
        if chunk_idx == 0:
            prefill = True
            chunk_prefill = False
        elif self.simulate_cache_pop and chunk_idx == self.round_boundary:
            # Simulate Round 2+ behavior
            prefill = False
            chunk_prefill = True  # <-- KEY CHANGE
            self._pop_kv_cache_to_sink()
        else:
            prefill = False
            chunk_prefill = False
```

### Config

```yaml
self_forcing:
  chunk_prefill_after_pop: true  # Enable chunk_prefill mode after cache pop
```

---

## Mismatch #2: KV Cache Behavior

### Problem

```
TRAINING:
  Cache evolution: [] → [F0] → [F0,F1] → [F0,F1,F2] → [F0,F1,F2,F3]
  → Monotonically growing, never pops

EVAL (Round 2+):
  Before pop: [F0,F1,F2,F3,F4] = 2300 tokens
  After pop:  [F0] = 460 tokens  ← ONLY SINK REMAINS
  After prefill: [F0,F1,F2,F3,F4] = 2300 tokens (rebuilt)
```

### Fix

```python
# Add to wrapper_causal.py

def _simulate_cache_pop(self, keep_sink_frames=1):
    """
    Simulate eval's cache pop: keep only first frame (sink).
    """
    sink_tokens = keep_sink_frames * self.block_size
    current_len = self.dit.kvcache_len()

    if current_len > sink_tokens:
        pop_amount = current_len - sink_tokens
        self.dit.pop_kvcache(pop_amount)
        return True  # Indicates chunk_prefill needed
    return False

def forward_self_forcing(self, ...):
    for chunk_idx in range(num_chunks):
        # At round boundary, simulate cache pop
        if chunk_idx == frames_per_round and self.config.simulate_cache_pop:
            needs_chunk_prefill = self._simulate_cache_pop(keep_sink_frames=1)
            if needs_chunk_prefill:
                chunk_prefill = True
```

### Config

```yaml
self_forcing:
  simulate_cache_pop: true
  sink_frames: 1              # Keep first frame after pop
  frames_per_round: 4         # Pop happens after this many frames
```

---

## Mismatch #3: RoPE Positioning

### Problem

```python
# Training uses rope_apply_one with absolute block_idx
rope_apply_one(x, grid_size, freqs, block_idx=chunk_idx, block_size=460)
# chunk_idx=2 → RoPE positions [920:1380]

# Eval uses rope_apply_chunk with relative kv_num_block
rope_apply_chunk(x, grid_size, freqs, kv_num_block=1, block_size=460)
# kv_num_block=1 (sink), x has 4 blocks → RoPE positions [460:2300]
# Frames get positions 1,2,3,4 relative to sink, NOT absolute 5,6,7,8
```

### Current RoPE Functions

```python
# rope_apply_one (training) - model_causal.py:91
if block_idx is not None:
    freqs_i = freqs_i[block_idx * block_size:(block_idx + 1) * block_size]

# rope_apply_chunk (eval after pop) - model_causal.py:130
if kv_num_block is not None:
    freqs_i = freqs_i[kv_num_block * block_size:(kv_num_block + x_num_block) * block_size]
```

### Fix

When `chunk_prefill=True`, the model automatically uses `rope_apply_chunk`. The fix is to ensure:
1. Training sets `chunk_prefill=True` after simulated cache pop
2. Pass correct `kv_num_block` (number of blocks remaining after pop = 1 for sink)

```python
# In forward_self_forcing after cache pop
if chunk_prefill:
    # Model will use rope_apply_chunk internally
    # kv_num_block is derived from current cache length
    kv_num_block = self.dit.kvcache_len() // self.block_size  # = 1 (sink only)
```

No explicit code change needed if Mismatch #1 and #2 are fixed - the model handles it internally.

---

## Mismatch #4: Attention Mask

### Problem

```python
# Training: Simple causal or chunk causal
attention_mask = self._build_chunk_causal_mask(...)  # Q attends to all cached KV

# Eval: Sink + causal within new frames (block_attention.py:53)
def block_mask_mod(b, h, q_idx, kv_idx):
    bqi = q_idx // block_size
    bki = kv_idx // block_size
    if bki < num_kvblock:  # Sink region
        return True        # Always attend to sink
    else:                  # New frames region
        return bqi + num_kvblock >= bki  # Causal within new
```

**Resulting mask pattern:**
```
       KV:  [F0 sink] [F1] [F2] [F3] [F4]
Q: [F1]       ✓        ✓    ✗    ✗    ✗
   [F2]       ✓        ✓    ✓    ✗    ✗
   [F3]       ✓        ✓    ✓    ✓    ✗
   [F4]       ✓        ✓    ✓    ✓    ✓
```

### Fix

```python
# In wrapper_causal.py

def _build_chunk_prefill_mask(self, num_new_frames, num_sink_blocks=1):
    """
    Build attention mask matching eval's chunk_prefill pattern.
    """
    from vidar.wan.modules.block_attention import get_flex_block_mask_chunk_prefill
    return get_flex_block_mask_chunk_prefill(
        num_qblock=num_new_frames,
        num_kvblock=num_sink_blocks,  # Sink blocks in cache
        block_size=self.block_size,
        device=self.device
    )

def forward_self_forcing(self, ...):
    if chunk_prefill:
        attention_mask = self._build_chunk_prefill_mask(
            num_new_frames=chunk_size,
            num_sink_blocks=1
        )
```

---

## Mismatch #5: Max Visible Context

### Problem

```
Training: Frame N can see [F0, F1, F2, ..., F(N-1)] = N frames (unbounded)
Eval:     Frame N can see [F0, F(N-4), F(N-3), F(N-2), F(N-1)] = 5 frames (bounded)
          (sink + last 4 conditioning frames)
```

### Fix

This is automatically handled by Mismatch #2 (cache pop). After pop:
- Only sink (F0) remains in cache
- New conditioning frames (last 4) are re-prefilled
- Total visible = 1 sink + 4 new = 5 frames

No additional code needed if cache pop is implemented correctly.

### Verification

```python
# Add logging to verify max context
def forward_self_forcing(self, ...):
    if self.debug:
        visible_frames = self.dit.kvcache_len() // self.block_size
        print(f"Frame {chunk_idx}: visible_context={visible_frames} frames")
        assert visible_frames <= 8, f"Context too large: {visible_frames}"
```

---

## Implementation Roadmap

### Phase 1: Core Changes (P0)

```
┌──────────────────────────────────────────────────────────────┐
│  1. Add cache pop simulation                                  │
│     File: wrapper_causal.py                                   │
│     Function: _simulate_cache_pop()                           │
│                                                               │
│  2. Enable chunk_prefill after pop                            │
│     File: wrapper_causal.py                                   │
│     Function: forward_self_forcing()                          │
│     Change: chunk_prefill = True after pop                    │
│                                                               │
│  3. Use correct attention mask                                │
│     File: wrapper_causal.py                                   │
│     Function: _build_chunk_prefill_mask()                     │
│     Import: get_flex_block_mask_chunk_prefill                 │
└──────────────────────────────────────────────────────────────┘
```

### Phase 2: Config & Testing (P1)

```yaml
# vidarc_aligned.yaml
self_forcing:
  enabled: true

  # Cache pop simulation (Mismatch #2)
  simulate_cache_pop: true
  sink_frames: 1
  frames_per_round: 4

  # Chunk prefill (Mismatch #1, #3, #4)
  chunk_prefill_after_pop: true

  # Context limit verification (Mismatch #5)
  max_context_frames: 8

  # Existing
  chunk_size: 1
  num_inference_steps: 10
```

---

## Code Change Summary

### File: `wrapper_causal.py`

```python
# Line ~28: Add import
from vidar.wan.modules.block_attention import get_flex_block_mask_chunk_prefill

# Line ~380: Add helper method
def _simulate_cache_pop(self, keep_sink_frames=1):
    sink_tokens = keep_sink_frames * self.block_size
    current_len = self.dit.kvcache_len()
    if current_len > sink_tokens:
        self.dit.pop_kvcache(current_len - sink_tokens)
        return True
    return False

def _build_chunk_prefill_mask(self, num_new_frames, num_sink_blocks=1):
    return get_flex_block_mask_chunk_prefill(
        num_qblock=num_new_frames,
        num_kvblock=num_sink_blocks,
        block_size=self.block_size,
        device=self.device
    )

# Line ~389-430: Modify forward_self_forcing
def forward_self_forcing(self, ...):
    frames_per_round = self.config.get('frames_per_round', 4)

    for chunk_idx in range(num_chunks):
        if chunk_idx == 0:
            prefill = True
            chunk_prefill = False
            attention_mask = self._build_causal_mask(...)

        elif chunk_idx == frames_per_round and self.config.simulate_cache_pop:
            # Round 2 starts: simulate cache pop
            self._simulate_cache_pop(keep_sink_frames=1)
            prefill = False
            chunk_prefill = True  # KEY: Enable chunk_prefill
            attention_mask = self._build_chunk_prefill_mask(
                num_new_frames=frames_per_round,
                num_sink_blocks=1
            )

        else:
            prefill = False
            chunk_prefill = False
            attention_mask = self._build_chunk_causal_mask(...)

        # Forward pass with correct flags
        output = self.forward_causal(
            ...,
            chunk_prefill=chunk_prefill,
            attention_mask=attention_mask,
        )
```

---

## Validation Checklist

After implementation, verify:

- [ ] `chunk_prefill=True` appears in training logs after round boundary
- [ ] Cache pop: `kvcache_len` drops to 460 (1 frame) at round boundary
- [ ] RoPE uses `kv_num_block=1` after pop (check `rope_apply_chunk` calls)
- [ ] Attention mask uses `get_flex_block_mask_chunk_prefill`
- [ ] Max visible context ≤ 8 frames throughout training
- [ ] Training loss curve is stable (no explosion after changes)
- [ ] Eval performance improves (key success metric)

### Debug Commands

```bash
# Verify chunk_prefill usage
grep "chunk_prefill=True" log_train.log

# Verify cache pop
grep "pop_kvcache\|After pop\|kvcache_len" log_train.log

# Verify RoPE mode
grep "rope_apply_chunk\|kv_num_block" log_train.log

# Compare with eval
diff <(grep "chunk_prefill\|kv_num_block" log_train.log) \
     <(grep "chunk_prefill\|kv_num_block" log_eval.log)
```

---

## Expected Behavior After Fix

```
TRAINING (aligned):
  Round 1:
    chunk_idx=0: prefill=True, chunk_prefill=False
    chunk_idx=1: cache=True, block_idx=1
    chunk_idx=2: cache=True, block_idx=2
    chunk_idx=3: cache=True, block_idx=3

  Round 2 (cache pop at chunk_idx=4):
    → pop_kvcache() called, only F0 remains
    chunk_idx=4-7: chunk_prefill=True, kv_num_block=1
    attention_mask=get_flex_block_mask_chunk_prefill(4, 1, 460)

EVAL (unchanged):
  Round 1: Same as training Round 1
  Round 2: chunk_prefill=True, kv_num_block=1

→ MATCH! Training and eval now use identical patterns.
```

---

## Debug Print Plan

### Location 1: `wrapper_causal.py` - forward_self_forcing()

Add at the start of each chunk iteration:

```python
def forward_self_forcing(self, x_clean, context, chunk_size=16, ...):
    # ... setup code ...

    for chunk_idx in range(num_chunks):
        # DEBUG: Print chunk state
        kv_len = self.dit.kvcache_len() if self.dit.kvcache_len() else 0
        print(f"[TRAIN] chunk_idx={chunk_idx}, kvcache_len={kv_len}, "
              f"kv_frames={kv_len // self.block_size}")

        if chunk_idx == 0:
            prefill = True
            chunk_prefill = False
            print(f"[TRAIN] Mode: PREFILL (first chunk)")

        elif chunk_idx == frames_per_round and self.config.simulate_cache_pop:
            # Cache pop
            kv_before = self.dit.kvcache_len()
            self._simulate_cache_pop(keep_sink_frames=1)
            kv_after = self.dit.kvcache_len()
            print(f"[TRAIN] CACHE POP: {kv_before} -> {kv_after} tokens "
                  f"({kv_before // self.block_size} -> {kv_after // self.block_size} frames)")

            prefill = False
            chunk_prefill = True
            print(f"[TRAIN] Mode: CHUNK_PREFILL (after pop)")

        else:
            prefill = False
            chunk_prefill = False
            print(f"[TRAIN] Mode: CACHE (block_idx={chunk_idx})")

        # After forward
        print(f"[TRAIN] Forward: prefill={prefill}, chunk_prefill={chunk_prefill}, "
              f"block_idx={None if prefill else chunk_idx}")
```

### Location 2: `model_causal.py` - Attention.forward()

Add to distinguish RoPE paths:

```python
def forward(self, x, grid_sizes, freqs, attention_mask=None, block_size=None,
            cache=False, prefill=False, chunk_prefill=False, block_idx=None):

    q, k, v = self.qkv_fn(x)

    if chunk_prefill:
        # DEBUG: chunk_prefill path
        cur_kv_len = self.cached_k.shape[1] if self.cached_k is not None else 0
        cur_kv_num_block = cur_kv_len // block_size
        print(f"[ATTN] chunk_prefill=True, kv_num_block={cur_kv_num_block}, "
              f"q_blocks={x.shape[1] // block_size}")
        return self.chunk_prefill(q, k, v, attention_mask, grid_sizes, freqs, block_size)

    # DEBUG: rope_apply_one path
    print(f"[ATTN] chunk_prefill=False, block_idx={block_idx}, prefill={prefill}")
    q = rope_apply_one(q, grid_sizes[0], freqs, block_idx=block_idx, block_size=block_size)
    k = rope_apply_one(k, grid_sizes[0], freqs, block_idx=block_idx, block_size=block_size)
    # ... rest of forward
```

### Location 3: `model_causal.py` - rope_apply_chunk()

```python
def rope_apply_chunk(x, grid_size, freqs_in, kv_num_block=None, block_size=None):
    x_num_block = x.shape[1] // block_size
    print(f"[ROPE_CHUNK] kv_num_block={kv_num_block}, x_num_block={x_num_block}, "
          f"positions=[{kv_num_block * block_size}:{(kv_num_block + x_num_block) * block_size}]")
    # ... rest of function
```

### Location 4: `block_attention.py` - get_flex_block_mask_chunk_prefill()

Already has debug prints, ensure they are enabled:

```python
def get_flex_block_mask_chunk_prefill(num_qblock, num_kvblock, block_size, device):
    Q_LEN = num_qblock * block_size
    KV_LEN = (num_kvblock + num_qblock) * block_size
    print(f"[MASK] chunk_prefill mask: num_qblock={num_qblock}, num_kvblock={num_kvblock}, "
          f"Q_LEN={Q_LEN}, KV_LEN={KV_LEN}")
    # ... rest of function
```

---

## Expected Debug Output

### Training (After Fix) - 9 latent frames, 2 rounds

```
=== Round 1 ===
[TRAIN] chunk_idx=0, kvcache_len=0, kv_frames=0
[TRAIN] Mode: PREFILL (first chunk)
[TRAIN] Forward: prefill=True, chunk_prefill=False, block_idx=None
[ATTN] chunk_prefill=False, block_idx=None, prefill=True

[TRAIN] chunk_idx=1, kvcache_len=460, kv_frames=1
[TRAIN] Mode: CACHE (block_idx=1)
[TRAIN] Forward: prefill=False, chunk_prefill=False, block_idx=1
[ATTN] chunk_prefill=False, block_idx=1, prefill=False

[TRAIN] chunk_idx=2, kvcache_len=920, kv_frames=2
[TRAIN] Mode: CACHE (block_idx=2)
[TRAIN] Forward: prefill=False, chunk_prefill=False, block_idx=2
[ATTN] chunk_prefill=False, block_idx=2, prefill=False

[TRAIN] chunk_idx=3, kvcache_len=1380, kv_frames=3
[TRAIN] Mode: CACHE (block_idx=3)
[TRAIN] Forward: prefill=False, chunk_prefill=False, block_idx=3
[ATTN] chunk_prefill=False, block_idx=3, prefill=False

=== Round 2 (cache pop at chunk_idx=4) ===
[TRAIN] chunk_idx=4, kvcache_len=1840, kv_frames=4
[TRAIN] CACHE POP: 1840 -> 460 tokens (4 -> 1 frames)
[TRAIN] Mode: CHUNK_PREFILL (after pop)
[TRAIN] Forward: prefill=False, chunk_prefill=True, block_idx=4
[ATTN] chunk_prefill=True, kv_num_block=1, q_blocks=4
[ROPE_CHUNK] kv_num_block=1, x_num_block=4, positions=[460:2300]
[MASK] chunk_prefill mask: num_qblock=4, num_kvblock=1, Q_LEN=1840, KV_LEN=2300

[TRAIN] chunk_idx=5, kvcache_len=2300, kv_frames=5
[TRAIN] Mode: CACHE (block_idx=5)
...

[TRAIN] chunk_idx=8, kvcache_len=3680, kv_frames=8
[TRAIN] Mode: CACHE (block_idx=8)
```

### Eval (Reference) - Should Match Round 2 Pattern

```
=== Round 1 ===
[EVAL] prefill: kvcache_len=0, T=1
[EVAL] block_idx=None, prefill=True

[EVAL] block_idx=1, cache=True
[EVAL] block_idx=2, cache=True
[EVAL] block_idx=3, cache=True
[EVAL] block_idx=4, cache=True (end of round 1, kvcache_len=2300)

=== Round 2 ===
[EVAL] prefill: kvcache_len=2300, T=4, T*block_size=1840
[EVAL] Using chunk prefill path (kvcache_len=2300 > T*block_size=1840)
[EVAL] CACHE POP: will pop 1840 from cache
[EVAL] After pop, kvcache_len=460, num_kvblock=1
[EVAL] chunk_prefill=True, Q_LEN=1840, KV_LEN=2300
[MASK] chunk_prefill mask: num_qblock=4, num_kvblock=1, Q_LEN=1840, KV_LEN=2300
[ROPE_CHUNK] kv_num_block=1, x_num_block=4, positions=[460:2300]
```

---

## Verification: Matching Patterns

After implementing, run both training and eval, then compare:

```bash
# Extract key patterns from both logs
grep -E "\[TRAIN\]|\[ATTN\]|\[ROPE_CHUNK\]|\[MASK\]" log_train.log > train_patterns.txt
grep -E "\[EVAL\]|\[ATTN\]|\[ROPE_CHUNK\]|\[MASK\]" log_eval.log > eval_patterns.txt

# Key checks:
# 1. Both should show chunk_prefill=True after round boundary
grep "chunk_prefill=True" train_patterns.txt eval_patterns.txt

# 2. Both should show kv_num_block=1 after cache pop
grep "kv_num_block=1" train_patterns.txt eval_patterns.txt

# 3. Both should show same Q_LEN, KV_LEN for chunk_prefill mask
grep "Q_LEN=1840, KV_LEN=2300" train_patterns.txt eval_patterns.txt

# 4. Both should show same RoPE positions
grep "positions=\[460:2300\]" train_patterns.txt eval_patterns.txt
```

### Success Criteria

| Check | Training | Eval | Match? |
|-------|----------|------|--------|
| Cache pop triggers | `1840 -> 460` | `2300 -> 460` | ✓ (both keep sink) |
| chunk_prefill flag | `True` after pop | `True` after pop | ✓ |
| kv_num_block | `1` | `1` | ✓ |
| Q_LEN | `1840` (4 frames) | `1840` (4 frames) | ✓ |
| KV_LEN | `2300` (5 frames) | `2300` (5 frames) | ✓ |
| RoPE positions | `[460:2300]` | `[460:2300]` | ✓ |

---

## Appendix: Code References

| Mismatch | File | Line | Function |
|----------|------|------|----------|
| #1 chunk_prefill | `wrapper_causal.py` | 394, 402 | `forward_self_forcing` |
| #1 chunk_prefill | `textimage2video_causal_server.py` | 554 | prefill logic |
| #2 Cache pop | `model_causal.py` | 645-648 | `pop_kvcache` |
| #2 Cache pop | `textimage2video_causal_server.py` | 545 | `pop_kvcache` call |
| #3 RoPE one | `model_causal.py` | 91-126 | `rope_apply_one` |
| #3 RoPE chunk | `model_causal.py` | 130-157 | `rope_apply_chunk` |
| #4 Mask | `block_attention.py` | 53-85 | `get_flex_block_mask_chunk_prefill` |
| #5 Context | Derived from #2 | - | Automatic via cache pop |
