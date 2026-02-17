# Weekly Progress Report (Simplified)
**Week of:** February 3-7, 2026 | **Reading time:** ~10 minutes

---

## Executive Summary

| Achievement | Status |
|-------------|--------|
| **Training Speedup** | 2.5× faster (optimizations only, excluding GPU scaling) |
| **Subgoal System** | ✅ Implemented - hierarchical planning for long-horizon tasks |
| **MPC Integration** | ✅ Implemented - sampling-based trajectory optimization |
| **Keyframe Extraction** | ✅ 5 semantic strategies for subgoal generation |

---

## I. Training Speed Optimization (Without GPU Scaling)

### Key Optimizations Deployed

| Optimization | Impact | Details |
|--------------|--------|---------|
| **T5 Embedding Cache** | ~50% T5 time reduction | Cache repeated prompt embeddings |
| **T5 Encoder Compilation** | ~15% additional speedup | `torch.compile()` on T5 encoder |
| **TF32 Support** | ~10% matmul speedup | Native H100/H200 optimization |

### Results (Computational Only)

```
Before optimizations:  593s/step
After optimizations:   ~237s/step (2.5× faster)
```

**Bottleneck Analysis:**
- Original: T5 encoding consumed 61% of step time (362.7s)
- After: T5 reduced to ~4% of step time (~10s)

---

## II. Subgoal Implementation Progress

### Architecture Overview

```
Task Description → Subgoal Generator → [subgoal_1, ..., subgoal_n]
                                              ↓
                                         Wan + IDM
                                              ↓
                                     Video + Actions (per subgoal)
```

### Key Design Points

1. **Input:** Text embedding from T5 + observation context
2. **Output:** N intermediate keyframe representations (default N=4)
3. **Training:** End-to-end with auxiliary L2 loss (weight=0.1)
4. **Overhead:** ~0.8% of training time (minimal)

### Keyframe Extraction Strategies

Five strategies implemented for extracting ground-truth subgoals:

| Strategy | Description | HDF5 Required |
|----------|-------------|---------------|
| **Uniform** | Fixed interval (baseline) | No |
| **Visual Change** | MSE-based frame difference | No |
| **Gripper Change** ⭐ | Detect grasp/release events | Yes |
| **Action Milestone** | Motion state transitions | Yes |
| **Semantic** | Combined motion + visual | No |

**Recommendation:** Use **Gripper Change** for manipulation tasks - best semantic alignment.

---

## III. MPC Optimization Flow

### 4-Step Pipeline

```
Step 1: Sample L candidate trajectories (different seeds)
    ↓
Step 2: IDM extracts action sequences from each trajectory
    ↓
Step 3: Evaluate costs (task + control + reachability)
    ↓
Step 4: Select best action (lowest cost)
```

### Step 1: Candidate Sampling

```python
def _sample_candidate_trajectories(self, obs, prompt, num_candidates=10, horizon=16):
    candidates = []
    for l in range(num_candidates):
        seed_l = np.random.randint(0, 2**31)

        frames = self.wan_model.generate(
            input_prompt=prompt,
            img=obs,
            num_new_frames=horizon,
            seed=seed_l,           # Different seed → different future
            clean_cache=True
        )
        candidates.append({'seed': seed_l, 'frames': frames})
    return candidates
```

**Key insight:** Different random seeds explore diverse possible futures.

### Step 2: IDM Action Extraction

```python
def _extract_action_sequences(self, candidates):
    action_sequences = []
    for candidate in candidates:
        frames = candidate['frames']
        actions = []

        # Chain backwards: IDM predicts action that led to each frame
        for i in range(len(frames) - 1, -1, -1):
            action = self.idm_model(self.processor(frames[i]))
            actions.insert(0, action)

        action_sequences.append(actions)
    return action_sequences
```

### Step 3: Cost Evaluation

```python
def _evaluate_costs(self, candidates, action_sequences):
    costs = []
    for l in range(len(candidates)):
        # Three cost components
        c_task  = self._compute_task_cost(...)      # Goal achievement
        c_ctrl  = self._compute_control_cost(...)   # Smoothness + magnitude
        c_reach = self._compute_reachability_cost(...)  # IK feasibility

        # Weighted sum
        J = λ_task * c_task + λ_ctrl * c_ctrl + λ_reach * c_reach
        costs.append(J)
    return costs
```

**Cost Components:**

| Cost | Formula | Purpose |
|------|---------|---------|
| Task | `‖final_frame - goal‖` | Achieve objective |
| Control | `0.5 * smoothness + 0.5 * magnitude` | Avoid jerky motion |
| Reachability | `+1.0 per infeasible action` | Ensure IK valid |

### Step 4: Action Selection

```python
def optimize(self, obs, prompt):
    candidates = self._sample_candidate_trajectories(obs, prompt)
    action_sequences = self._extract_action_sequences(candidates)
    costs = self._evaluate_costs(candidates, action_sequences)

    best_idx = np.argmin(costs)
    return action_sequences[best_idx][0]  # Execute first action only
```

### MPC Workflow Diagram

```
                    ┌─────────────────────────────┐
                    │     Current Observation     │
                    └─────────────┬───────────────┘
                                  ↓
              ┌───────────────────┴───────────────────┐
              │    Sample L=10 candidate futures      │
              │    (Wan with different seeds)         │
              └───────────────────┬───────────────────┘
                                  ↓
              ┌───────────────────┴───────────────────┐
              │    IDM: Extract action sequences      │
              │    frames → [a₀, a₁, ..., aₕ₋₁]       │
              └───────────────────┬───────────────────┘
                                  ↓
              ┌───────────────────┴───────────────────┐
              │    Evaluate: J = λ·c_task + c_ctrl    │
              └───────────────────┬───────────────────┘
                                  ↓
              ┌───────────────────┴───────────────────┐
              │    Select: l* = argmin J              │
              │    Execute: a₀^{l*}                   │
              └───────────────────┬───────────────────┘
                                  ↓
                    ┌─────────────┴───────────────┐
                    │    Observe → Re-plan        │
                    └─────────────────────────────┘
```

### Default Configuration

```python
mpc_config = {
    'num_candidates': 10,      # L: number of samples
    'horizon': 16,             # H: prediction steps
    'cost_weights': {
        'task': 0.6,
        'ctrl': 0.3,
        'reach': 0.1
    }
}
```

---

## IV. Evaluation Framework

### Three Evaluation Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Direct** | End-to-end Wan+IDM | Baseline performance |
| **Hierarchical** | With learned subgoals | Test subgoal system |
| **Oracle** | With GT keyframes | Upper bound |

---

## V. Key Metrics Summary

### Training Performance

| Metric | Value |
|--------|-------|
| Step time (optimized) | 236s |
| T5 cache hit rate | ~70% |
| GPU memory | ~18GB/80GB |
| Loss variance | <0.1 |

### Expected Improvements

| Feature | Expected Impact |
|---------|-----------------|
| Hierarchical subgoals | +15-20% success rate on long-horizon tasks |
| MPC | +15-25% success rate vs direct policy |

---

## VI. Next Week's Plan (Feb 10-14)

### Priority Tasks

| Task | Duration | Deliverable |
|------|----------|-------------|
| **Subgoal validation** | 2-3 days | Tune loss weight, visualize learned vs GT |
| **MPC tuning** | 1-2 days | Optimal horizon, re-plan frequency |
| **Evaluation suite** | 2 days | Compare 3 modes at step 500, 1000, 2000 |
| **Few-step diffusion** | 1-2 days | 5-step vs 10-step quality/speed trade-off |
| **Documentation** | 1 day | API docs, configuration guide |

### Success Criteria

- Subgoal L2 error < 0.5 by step 2000
- No degradation in video generation quality
- Statistical significance (n=50 episodes per task)

### Open Research Questions

1. **Subgoal loss weight:** 0.1 vs 0.5 - impact on learning?
2. **Keyframe strategy:** Gripper vs uniform - which generalizes better?
3. **MPC horizon:** Optimal N for different task complexities?

---

## VII. Visual Summary

```
┌─────────────────────────────────────────────────────────┐
│  Training Speed:    593s → 237s  (2.5× computational)   │
│  New Features:      Subgoals ✅  MPC ✅  Eval ✅         │
│  ETA:               ~11 days (with 8 GPU scaling)       │
└─────────────────────────────────────────────────────────┘

Architecture:
┌─────────────────────────────────────────────┐
│         Fine-Tuning Enhancements            │
│   Subgoal Generator + MPC Controller [NEW]  │
└─────────────────┬───────────────────────────┘
                  ↓
┌─────────────────┴───────────────────────────┐
│         Base Models (Fine-Tuning)           │
│   Wan2.2-TI2V-5B + IDM (Pre-trained)        │
└─────────────────────────────────────────────┘
```

---

**Report Date:** February 7, 2026
