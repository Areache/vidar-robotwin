# Vidar Fine-Tuning Pipeline

Fine-tuning Vidar video diffusion model using Self-Forcing paradigm for embodied closed-loop control.

---

## Checkpoints

### Checkpoint Loading Architecture

```python
# From textimage2video.py / textimage2video_causal.py
self.model = WanModel.from_pretrained(checkpoint_dir)      # Load Wan2.2 base
if pt_dir is not None:
    self.model.load_state_dict(torch.load(pt_dir), strict=False)  # Apply fine-tuned weights
```

| Checkpoint | Model Class | Script | Content |
|------------|-------------|--------|---------|
| `Wan2.2-TI2V-5B/` | base | - | T5 encoder + VAE + DiT base weights |
| `vidar.pt` | WanModel | `generate.py` | Fine-tuned DiT weights (delta on Wan2.2) |
| `vidarc.pt` | WanModelCausal | `generate_causal.py` | Causal DiT weights (delta on Wan2.2) |
| `idm.pt` | IDM | separate | Inverse Dynamics Model (standalone) |

### Downloads

```bash
# Base model (required) - contains T5, VAE, DiT base
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir ./checkpoints/Wan2.2-TI2V-5B

# Fine-tuned weights
huggingface-cli download Xiang-cd/vidar --local-dir ./checkpoints/vidar
```

| File | Size | Loads Into |
|------|------|------------|
| `Wan2.2-TI2V-5B/` | ~15GB | WanModel / WanModelCausal (via `from_pretrained`) |
| `vidar.pt` | 10 GB | WanModel (via `load_state_dict`, `strict=False`) |
| `vidarc.pt` | 10 GB | WanModelCausal (via `load_state_dict`, `strict=False`) |
| `idm.pt` | 1.11 GB | IDM (separate model) |

**Demo**: https://huggingface.co/spaces/Wan-AI/Wan-2.2-5B

---

## Training Stages (Decoupled)

Both stages can be debugged **independently** since we have all checkpoints.

### Stage 1: Vidar Fine-tuning (Standard Diffusion)
```
Load:   WanModel.from_pretrained(Wan2.2-TI2V-5B/)  # Base DiT
Train:  WanModel (DiT only, freeze T5 & VAE)
Save:   model.state_dict() → vidar.pt
```
- **Steps**: 14k
- **Loss**: Flow matching
- **Model**: `WanModel`
- **Script**: `training/scripts/train_vidar.py`

### Stage 2: Vidarc Causal Training (Self-Forcing)
```
Load:   WanModelCausal.from_pretrained(Wan2.2-TI2V-5B/)
        + load_state_dict(vidar.pt, strict=False)
Train:  WanModelCausal (with KV cache, autoregressive)
Save:   model.state_dict() → vidarc.pt
```
- **Steps**: 4k
- **Loss**: Causal flow matching + embodiment-aware
- **Model**: `WanModelCausal`
- **Script**: `training/scripts/train_vidarc.py`

### Parallel Debug Strategy
```bash
# Terminal 1: Debug Stage 1 (WanModel)
python training/scripts/train_vidar.py \
    --ckpt_dir checkpoints/Wan2.2-TI2V-5B \
    --debug --max_steps 100

# Terminal 2: Debug Stage 2 (WanModelCausal) - uses existing vidar.pt
python training/scripts/train_vidarc.py \
    --ckpt_dir checkpoints/Wan2.2-TI2V-5B \
    --pt_dir checkpoints/vidar/vidar.pt \
    --debug --max_steps 100
```

### Key Difference: WanModel vs WanModelCausal
| | WanModel | WanModelCausal |
|-|----------|----------------|
| Attention | Bidirectional | Causal (frame-by-frame) |
| KV Cache | No | Yes |
| Inference | Batch all frames | Autoregressive |
| File | `model.py` | `model_causal.py` |

---

## Hyperparameters (from paper Table 5)

| Param | Vidar/Vidarc | IDM |
|-------|--------------|-----|
| Parameters | 5B | 92M |
| LR | 2e-5 | 5e-4 |
| Batch size | 128 | 128 |
| Warmup | 200 steps | 6k steps |
| Optimizer | AdamW (β=0.9,0.999, ε=1e-8, wd=0.1) | AdamW (wd=1e-2) |

**Data config**: 81 frames @ 10fps, resolution 736×640, CFG prob=0.1

---

## Loss Functions

### Flow Matching (Eq. 1)
```python
L = ||v_θ(x_t, t, c) - (x_0 - x_1)||²
# x_t = t*x_1 + (1-t)*x_0
```

### Causal (Eq. 6)
```python
L = ||v_θ(x_t, t, c, x_prev) - (x_0 - x_1)||²
# x_prev: noise-free previous frames via KV cache
```

### Embodiment-Aware (Eq. 7)
```python
L = ||(1 + η·U(x_1)) ⊙ (v_θ - (x_0 - x_1))||²
# U(x_1): IDM mask, η=3.0
```

### IDM (Eq. 4)
```python
L = huber(â - a) + λ·||m||₁  # λ=3e-3
```

---

## Self-Forcing Training

Simulates inference during training with autoregressive rollout + KV caching:

```python
def train_step(model, instruction, video_frames):
    kv_cache = model.prefill(video_frames[0], instruction)
    total_loss = 0

    for t in range(1, chunk_size):
        timestep = sample_timestep()
        noised = add_noise(video_frames[t], timestep)

        # Forward with KV cache (matches inference)
        pred, kv_cache = model.forward_with_cache(noised, timestep, kv_cache)

        # Embodiment-aware loss
        mask = idm.get_mask(video_frames[t])
        loss = embodiment_aware_loss(pred, video_frames[t], mask, eta=3.0)
        total_loss += loss

    return total_loss / chunk_size
```

---

## Config Files

### `configs/vidar_finetune.yaml` (Stage 1)
```yaml
model:
  ckpt_dir: checkpoints/Wan2.2-TI2V-5B  # Base model (T5 + VAE + DiT)
  pt_dir: null                           # No fine-tuned weights
  model_class: WanModel

training:
  num_steps: 14000
  batch_size: 128
  lr: 2.0e-5
  warmup: 200
  freeze: [t5, vae]                      # Only train DiT

loss:
  type: flow_matching
  embodiment_aware: false

output:
  save_path: checkpoints/vidar/vidar.pt  # Save DiT state_dict only
```

### `configs/vidarc_causal.yaml` (Stage 2)
```yaml
model:
  ckpt_dir: checkpoints/Wan2.2-TI2V-5B  # Base model
  pt_dir: checkpoints/vidar/vidar.pt    # Fine-tuned weights from Stage 1
  model_class: WanModelCausal

training:
  num_steps: 4000
  batch_size: 128
  lr: 2.0e-5
  freeze: [t5, vae]

loss:
  type: causal_flow_matching
  embodiment_aware: true
  eta: 3.0

self_forcing:
  causal: true
  chunk_size: 16
  kv_cache_length: 64

output:
  save_path: checkpoints/vidar/vidarc.pt
```

---

## Directory Structure

```
vidar-robotwin/
├── training/
│   ├── config.py              # TrainingConfig dataclass
│   ├── losses.py              # Loss implementations
│   ├── trainers/
│   │   ├── base.py
│   │   ├── vidar_trainer.py   # Stage 1
│   │   └── self_forcing.py    # Stage 2
│   ├── data/
│   │   ├── dataset.py         # VidarDataset
│   │   ├── transforms.py
│   │   └── observation.py     # Unified obs space (720×640)
│   ├── models/
│   │   ├── wrapper.py         # WanModel training wrapper
│   │   └── causal_wrapper.py  # WanModelCausal wrapper
│   ├── distributed/
│   │   └── fsdp_utils.py
│   └── scripts/
│       ├── train.py
│       └── train_vidarc.sh
├── configs/
│   ├── vidar_finetune.yaml
│   └── vidarc_causal.yaml
└── checkpoints/
    ├── Wan2.2-TI2V-5B/          # from_pretrained() loads this
    │   ├── config.json
    │   ├── diffusion_pytorch_model.safetensors
    │   ├── t5/
    │   └── vae/
    └── vidar/                    # pt_dir loads from here
        ├── vidar.pt              # WanModel fine-tuned weights
        ├── vidarc.pt             # WanModelCausal fine-tuned weights
        └── idm.pt                # IDM standalone
```

---

## Launch Training

### Single Node (8 GPUs)
```bash
torchrun --nproc_per_node=8 \
    training/scripts/train.py \
    --config configs/vidarc_causal.yaml \
    --output_dir outputs/vidarc
```

### Multi-Node (8×8=64 GPUs)
```bash
# Node 0
torchrun --nnodes=8 --nproc_per_node=8 --node_rank=0 \
    --rdzv_endpoint=$MASTER:29500 \
    training/scripts/train.py --config configs/vidarc_causal.yaml

# Node N
torchrun --nnodes=8 --nproc_per_node=8 --node_rank=$N \
    --rdzv_endpoint=$MASTER:29500 \
    training/scripts/train.py --config configs/vidarc_causal.yaml
```

---

## Hardware Requirements

| Config | Hardware | Time |
|--------|----------|------|
| Full | 64× H100 | ~2h |
| Grad accum | 8× H100 | ~16h |
| Single node | 8× A100-80G | ~24h |
| **Your setup** | **2× H200-120G** | **~48h** |

### 2×H200 Configuration

```yaml
# configs/vidarc_2xh200.yaml
training:
  batch_size: 16                    # 8 per GPU
  gradient_accumulation: 8          # Effective batch = 128
  num_steps: 4000

distributed:
  use_fsdp: true
  sharding_strategy: FULL_SHARD     # Max memory efficiency
  mixed_precision: bf16
  activation_checkpointing: true
  cpu_offload: false                # Enable if OOM

model:
  gradient_checkpointing: true
  chunk_size: 8                     # Reduce if memory tight (default 16)
```

```bash
# Launch on 2×H200
torchrun --nproc_per_node=2 \
    training/scripts/train.py \
    --config configs/vidarc_2xh200.yaml \
    --output_dir outputs/vidarc
```

**Memory estimate (5B model, bf16):**
- Model params: ~10GB
- Optimizer states: ~40GB (sharded across 2 GPUs = 20GB/GPU)
- Activations: ~40-60GB/GPU (with grad checkpointing)
- **Total: ~70-90GB/GPU** ✓ fits in 120GB

**If OOM**: reduce `chunk_size` to 4, or enable `cpu_offload: true`

---

## Dataset Format

```
data/
├── hdf5/
│   ├── episode_000000.hdf5
│   ├── episode_000001.hdf5
│   └── ...
├── config.json
├── instructions.json
└── episodes.json
```

### HDF5 Episode Structure
```
episode_XXXXXX.hdf5
├── observations/
│   ├── unified_image: (T, 720, 640, 3)    # Pre-composed multi-view
│   ├── images/
│   │   ├── cam_high: (T, H, W, 3)
│   │   ├── cam_left_wrist: (T, H, W, 3)
│   │   └── cam_right_wrist: (T, H, W, 3)
│   └── qpos: (T, 14)                       # Joint positions
├── action: (T, 14)                         # Action sequence
└── instruction (attr): string              # Language instruction
```

### Unified Observation Layout (720×640)
```
┌─────────────────────────────┐
│      cam_high (360×640)     │  ← Fixed front/rear camera
├──────────────┬──────────────┤
│ cam_left     │ cam_right    │
│ (360×320)    │ (360×320)    │  ← Movable arm cameras
└──────────────┴──────────────┘
```

---

## Datasets

### Overview (from Vidarc Paper Table 4)

| Dataset | Episodes | Type | Camera Setup | Use |
|---------|----------|------|--------------|-----|
| **Egodex** | 230,949 | Human hands | Movable front | Pre-train |
| **Agibot** | 728,209 | Genie-1 Robot | Fixed high + left/right arm | Pre-train |
| **RDT** | 6,083 | Aloha Robot | Fixed front + left/right arm | Pre-train |
| **RoboMind Franka** | 9,589 | Franka Robot | Fixed opposite + left/right | Pre-train |
| **RoboMind Aloha** | 7,272 | Aloha Robot | Fixed front + left/right arm | Pre-train |
| **RoboTwin** | 1,000 | Aloha Robot (Sim) | Fixed rear + left/right arm | **Fine-tune** |
| **Vidarc** | 2,307 | Aloha Robot (Real) | Fixed rear + left/right arm | **Fine-tune** |

### Open-Source Availability

| Dataset | Open Source | License | Download |
|---------|-------------|---------|----------|
| **Egodex** | ✅ Yes | CC-BY-NC-ND | [GitHub](https://github.com/apple/ml-egodex) |
| **Agibot World** | ✅ Yes | CC BY-NC-SA 4.0 | [HuggingFace](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta) |
| **RDT ft-data** | ✅ Yes | - | [HuggingFace](https://huggingface.co/datasets/robotics-diffusion-transformer/rdt-ft-data) |
| **RoboMIND** | ✅ Yes | - | [HuggingFace](https://huggingface.co/datasets/x-humanoid-robomind/RoboMIND) |
| **RoboTwin** | ✅ Yes | MIT | [GitHub](https://github.com/RoboTwin-Platform/RoboTwin) |
| **Vidarc real-world** | ❌ Not yet | - | [thu-ml/vidar](https://github.com/thu-ml/vidar) (planned) |

### Download Commands

```bash
# ============================================================
# Pre-training Datasets (~1M episodes total)
# ============================================================

# 1. Agibot World Beta (~1M trajectories, ~43.8TB) - LARGEST
huggingface-cli download --resume-download --repo-type dataset \
    agibot-world/AgiBotWorld-Beta --local-dir ./data/AgiBotWorld-Beta

# Or Alpha subset (~92k trajectories, ~8.5TB) - smaller
huggingface-cli download --resume-download --repo-type dataset \
    agibot-world/AgiBotWorld-Alpha --local-dir ./data/AgiBotWorld-Alpha

# 2. RDT Fine-tuning data (~6k Aloha episodes)
huggingface-cli download --resume-download --repo-type dataset \
    robotics-diffusion-transformer/rdt-ft-data --local-dir ./data/rdt-ft-data

# 3. RoboMIND (~107k trajectories across 4 robots)
huggingface-cli download --resume-download --repo-type dataset \
    x-humanoid-robomind/RoboMIND --local-dir ./data/RoboMIND

# 4. Egodex (829 hours egocentric video from Apple Vision Pro)
# See: https://github.com/apple/ml-egodex

# ============================================================
# Fine-tuning Datasets (1-2k episodes for Stage 2)
# ============================================================

# 5. RoboTwin Simulation Benchmark
git clone https://github.com/RoboTwin-Platform/RoboTwin.git ./data/RoboTwin
# Data: https://huggingface.co/datasets/robotwin (100k+ trajectories)
```

### Data Preparation Scripts

```bash
# ============================================================
# Prepare RDT data for training
# ============================================================
# Extract split archive
cd data/rdt-ft-data
cat rdt_data.tar.gz.* | tar -xzf -
cd ../..

# Convert to unified format
python scripts/prepare_rdt_data.py \
    --src-dir ./data/rdt-ft-data/rdt_data \
    --dst-dir ./data/rdt_prepared \
    --num-frames 81

# ============================================================
# Prepare RoboTwin data for Stage 2
# ============================================================
python scripts/prepare_vidarc_data.py \
    --src-dir ./data/RoboTwin/data \
    --dst-dir ./data/robotwin_prepared \
    --dataset-type robotwin \
    --num-frames 81

# ============================================================
# Prepare Agibot data
# ============================================================
python scripts/prepare_agibot_rdt.py \
    --src-dir ./data/AgiBotWorld-Alpha \
    --dst-dir ./data/agibot_prepared \
    --num-frames 81
```

### Instruction Format (Unified Observation Space)

All datasets use scene-description formatted instructions:

```python
# Template for Aloha robots (RoboTwin, RDT, Vidarc)
"The whole scene is in a realistic, industrial art style with three views: "
"a fixed rear camera, a movable left arm camera, and a movable right arm camera. "
"The aloha robot is currently performing the following task: {task_instruction}"

# Template for Agibot
"The whole scene is in a realistic, industrial art style with three views: "
"a fixed high camera, a movable left arm camera, and a movable right arm camera. "
"The genie-1 robot is currently performing the following task: {task_instruction}"
```

### Recommended Setup (2×H200)

For **Stage 1** (Vidar fine-tuning):
- Use subset of pre-training data or RDT + RoboMind (~20k episodes)
- 14k training steps

For **Stage 2** (Vidarc causal fine-tuning):
- Only need 1-2k episodes (RoboTwin or RDT subset)
- 4k training steps
- Starts from Stage 1 weights (`vidar.pt`)

```bash
# Minimal Stage 2 setup with RDT data
python scripts/prepare_rdt_data.py \
    --src-dir ./data/rdt-ft-data/rdt_data \
    --dst-dir ./data/vidarc_finetune \
    --num-frames 81 \
    --max-episodes 2000  # Use subset for faster iteration
```

### Data Specifications

| Parameter | Value | Notes |
|-----------|-------|-------|
| Resolution | 720×640 | Unified observation (H×W) |
| FPS | 10 | Downsampled from source |
| Frames/episode | 81 | 8.1 seconds duration |
| Action dim | 14 | Bimanual joint positions |
| State dim | 14 | Joint positions (qpos) |
| CFG dropout | 0.1 | Drop instruction 10% of time |

---

## Implementation Priority (Parallel Tracks)

### Shared (do first)
| Task | Files |
|------|-------|
| Config dataclass | `config.py` |
| Dataset & transforms | `data/dataset.py`, `data/transforms.py` |
| Base trainer | `trainers/base.py` |
| FSDP utils | `distributed/fsdp_utils.py` |

### Track A: Stage 1 (Vidar)
| Task | Files |
|------|-------|
| Flow matching loss | `losses.py` |
| WanModel wrapper | `models/wrapper.py` |
| Vidar trainer | `trainers/vidar_trainer.py` |
| Train script | `scripts/train_vidar.py` |

### Track B: Stage 2 (Vidarc)
| Task | Files |
|------|-------|
| Embodiment-aware loss | `losses.py` |
| WanModelCausal wrapper + KV cache | `models/causal_wrapper.py` |
| Self-Forcing trainer | `trainers/self_forcing.py` |
| Train script | `scripts/train_vidarc.py` |

---

## Key Model Files

| Component | Path |
|-----------|------|
| WanModel | `vidar/wan/modules/model.py` |
| WanModelCausal | `vidar/wan/modules/model_causal.py` |
| VAE | `vidar/wan/modules/vae2_2.py` |
| T5 Encoder | `vidar/wan/modules/t5.py` |
| IDM | `vidar/idm/idm/idm.py` |
| Configs | `vidar/wan/configs/shared_config.py` |

---

## References

- Vidarc Paper: arXiv:2512.17661
- Self-Forcing: https://github.com/guandeh17/Self-Forcing
- Wan2.1: https://github.com/Wan-Video/Wan2.1
