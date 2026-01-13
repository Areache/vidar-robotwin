# Subgoal Implementation Versions

This directory contains version definitions for the subgoal implementation.

## Available Versions

### v0_original
- **Description**: Original implementation without subgoal support
- **Characteristics**:
  - `use_libero_subgoal = False`
  - No subgoal generation
  - No subgoal_frames passed to vidar
  - Standard video generation pipeline
- **Use case**: Baseline comparison, debugging

### v1_subgoal
- **Description**: Current implementation with LIBERO subgoal support
- **Characteristics**:
  - `use_libero_subgoal = True`
  - Generates subgoals using LIBERO video model
  - Passes subgoal_frames to vidar for guidance
  - Uses subgoal_guidance_scale = 0.5 for blending
- **Use case**: Current production version

## Usage

### Via Command Line

```bash
# Use v0_original (no subgoals)
python policy/AR/run_eval_ddp.py \
  --version v0_original \
  --task_config hd_clean \
  ...

# Use v1_subgoal (with subgoals)
python policy/AR/run_eval_ddp.py \
  --version v1_subgoal \
  --task_config hd_clean \
  ...
```

### Via Configuration

Add `version` to your `usr_args`:

```python
usr_args = {
    "version": "v0_original",  # or "v1_subgoal"
    # ... other args
}
model = AR(usr_args=usr_args, version=usr_args.get("version"))
```

### Verification

Run the verification script to check both versions:

```bash
cd /mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar-robotwin
python policy/AR/verify_versions.py --version both
```

## Adding New Versions

To add a new version, edit `subgoal_versions.json`:

```json
{
  "versions": {
    "v2_new_version": {
      "name": "v2_new_version",
      "description": "New version description",
      "use_subgoals": true,
      "robotwin_params": {
        "use_libero_subgoal": true,
        "subgoal_interval": 4
      },
      "vidar_params": {
        "subgoal_guidance_scale": 0.7
      }
    }
  }
}
```

Then register it in your code:

```python
from policy.AR.version_config import load_version_config
version_config = load_version_config("policy/AR/versions/subgoal_versions.json")
```

