# Video-to-Action Video Model 集成指南

## 概述

本文档说明如何将 `video-to-action-release` 库的 video model 生成的 subgoal frames 集成到 `vidar-robotwin` 的 AR policy pipeline 中。

## 当前 Pipeline 架构

### Vidar-Robotwin Pipeline
```
eval_policy.py
    ↓
eval() → AR.eval()
    ↓
AR.get_actions() → HTTP请求 → Vidar Server (causal_worker.sh)
    ↓
返回动作序列 → 执行动作 → 更新观察 → 循环
```

### Video-to-Action Pipeline
```
plan_lb.py
    ↓
LB_DP_Eval.eval_1_env()
    ├── 1. 渲染初始观察
    ├── 2. 视频预测循环
    │   ├── video_model.forward() → 生成未来帧 (subgoals)
    │   └── diffusion_policy.predict_action() → 生成动作
    └── 3. 执行动作并更新观察
```

## 集成方案

### 方案1: 在 AR Policy 内部集成（推荐）

**集成点**: 在 `AR.get_actions()` 方法中，调用 vidar server 之前先调用 video model

**优点**:
- 最小化代码修改
- 保持现有架构不变
- 易于控制集成逻辑

**实现步骤**:

1. **修改 `policy/AR/ar.py`**:
   - 在 `__init__` 中初始化 video model
   - 在 `get_actions()` 中，调用 vidar server 前先调用 video model 生成 subgoal frames
   - 将 subgoal frames 作为额外输入传递给 vidar server（需要修改 server 接口）

2. **修改 Vidar Server** (`vidar/server/causal_worker.sh` 或对应的 Python 服务):
   - 接受 subgoal frames 作为可选输入
   - 在生成动作时考虑 subgoal frames

### 方案2: 在 eval_policy.py 中集成

**集成点**: 在 `eval()` 函数中，调用 `AR.eval()` 之前生成 subgoal frames

**优点**:
- 更灵活的控制
- 可以独立管理 video model 的生命周期

**实现步骤**:

1. **修改 `script/eval_policy.py`**:
   - 在 `eval_policy()` 函数中初始化 video model
   - 在调用 `eval_func()` 之前，生成 subgoal frames
   - 将 subgoal frames 传递给 AR model

2. **修改 `policy/AR/ar.py`**:
   - 添加 `set_subgoals()` 方法接收 subgoal frames
   - 在 `get_actions()` 中使用 subgoal frames

### 方案3: 混合方案（最灵活）

**集成点**: 在 AR model 中集成，但通过配置控制是否启用

**实现**:
- 添加配置参数 `use_video_subgoals: bool`
- 如果启用，在 `get_actions()` 中调用 video model
- 如果禁用，使用原有逻辑

## 关键技术考虑

### 1. 图像格式转换

**问题**: 两个库使用的图像格式可能不同

**Vidar-Robotwin**:
- 输入: `(H, W, 3)` numpy array, BGR格式
- 编码: JPEG base64

**Video-to-Action**:
- 输入: `(1, 3, H, W)` torch tensor, RGB格式, 值域 [0, 1]
- 预处理: `imgs_preproc_simple_noCrop_v1`

**解决方案**:
```python
def convert_obs_to_video_input(obs_img):
    """
    将 vidar-robotwin 的观察转换为 video model 输入格式
    
    Args:
        obs_img: (H, W, 3) numpy array, BGR格式
    
    Returns:
        img_tensor: (1, 3, H, W) torch tensor, RGB格式, [0, 1]
    """
    # BGR to RGB
    img_rgb = obs_img[:, :, ::-1]
    
    # Resize to (128, 128) if needed
    img_resized = cv2.resize(img_rgb, (128, 128))
    
    # Normalize to [0, 1] and convert to tensor
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    return img_tensor

def convert_subgoal_to_vidar_format(subgoal_tensor):
    """
    将 video model 输出的 subgoal 转换为 vidar 可用的格式
    
    Args:
        subgoal_tensor: (1, 3, H, W) torch tensor, RGB格式, [0, 1]
    
    Returns:
        subgoal_img: (H, W, 3) numpy array, BGR格式, uint8
    """
    # Denormalize and convert to numpy
    img = (subgoal_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # RGB to BGR
    img_bgr = img[:, :, ::-1]
    
    return img_bgr
```

### 2. 模型加载和初始化

**Video Model 初始化**:
```python
from diffuser.libero.lb_video_model_utils import lb_get_video_model_gcp_v2
from diffuser.libero.lb_eval_utils import load_lb_diffusion

# 需要配置参数
video_model_args = {
    'vid_diffusion': {
        'checkpoint_path': '/path/to/video/model/checkpoint.pt',
        'device': 'cuda:0',
        # ... 其他参数
    }
}

video_model = lb_get_video_model_gcp_v2(**video_model_args['vid_diffusion'])
video_model.eval()
```

**注意事项**:
- Video model 需要特定的配置文件和 checkpoint
- 需要确保 CUDA 设备一致性
- 考虑模型加载的内存占用

### 3. 任务文本格式

**Vidar-Robotwin**:
- 使用完整的任务描述，带系统提示词
- 例如: "The whole scene is in a realistic, industrial art style... put the red mug on the left plate"

**Video-to-Action**:
- 使用简洁的任务描述
- 例如: "put the red mug on the left plate"

**解决方案**:
```python
def extract_task_description(instruction):
    """
    从 vidar 的 instruction 中提取简洁的任务描述
    """
    # 移除系统提示词部分
    if "The whole scene" in instruction:
        # 提取任务部分
        task_part = instruction.split("performing the following task: ")[-1]
        return task_part
    return instruction
```

### 4. Subgoal 生成策略

**参考 video-to-action 的策略**:
- `num_vid_pred_per_ep = 5`: 每个 episode 预测 5 个视频
- `use_vid_first_n_frames = 2`: 每个视频使用前 2 帧作为 subgoal
- `v_hzn = 7`: 每个视频预测 7 帧
- `eval_n_preds_betw_vframes = 5`: 每个视频帧之间预测 5 次动作

**集成建议**:
```python
class AR:
    def __init__(self, ...):
        # ... 现有初始化
        self.use_video_subgoals = usr_args.get("use_video_subgoals", False)
        if self.use_video_subgoals:
            self.video_model = self._init_video_model()
            self.subgoal_queue = []  # 存储生成的 subgoals
            self.current_subgoal_idx = 0
            self.num_vid_pred_per_ep = usr_args.get("num_vid_pred_per_ep", 5)
            self.use_vid_first_n_frames = usr_args.get("use_vid_first_n_frames", 2)
    
    def _generate_subgoals(self, current_obs, task_description):
        """
        生成 subgoal frames
        """
        # 转换观察格式
        img_tensor = convert_obs_to_video_input(current_obs)
        
        # 提取任务描述
        task_str = extract_task_description(task_description)
        
        # 预测视频
        with torch.no_grad():
            preds_video = self.video_model.forward(
                img_tensor.to(self.device),
                [task_str]
            )
        
        # preds_video: (1, 7, 3, 128, 128)
        # 提取前 N 帧作为 subgoals
        subgoals = []
        for i in range(self.use_vid_first_n_frames):
            subgoal_tensor = preds_video[0, i]  # (3, H, W)
            subgoal_img = convert_subgoal_to_vidar_format(subgoal_tensor.unsqueeze(0))
            subgoals.append(subgoal_img)
        
        return subgoals
    
    def get_actions(self):
        # 如果启用 video subgoals 且队列为空，生成新的 subgoals
        if self.use_video_subgoals:
            if len(self.subgoal_queue) == 0:
                current_obs = self.obs_cache[-1]  # 获取最新观察
                subgoals = self._generate_subgoals(current_obs, self.prompt)
                self.subgoal_queue.extend(subgoals)
            
            # 使用当前 subgoal
            current_subgoal = self.subgoal_queue.pop(0)
            # 将 subgoal 传递给 vidar server（需要修改 server 接口）
        
        # ... 原有逻辑
```

### 5. 内存和性能考虑

**内存管理**:
- Video model 通常较大（几GB），需要确保有足够 GPU 内存
- 考虑使用 `torch.no_grad()` 和 `model.eval()` 减少内存占用
- 可以延迟加载 video model，只在需要时加载

**性能优化**:
- Video model 推理较慢，考虑异步生成 subgoals
- 可以预先生成一批 subgoals，而不是每次调用都生成
- 考虑使用更小的图像分辨率（如果允许）

### 6. 依赖管理

**需要安装的依赖**:
- `video-to-action-release` 库的所有依赖
- 确保两个库的 PyTorch 版本兼容
- 确保 CUDA 版本兼容

**环境变量**:
```bash
# 可能需要设置
export PYTHONPATH=/path/to/video-to-action-release:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0  # 确保 GPU 可见性一致
```

### 7. 配置参数

**在 `deploy_policy.yml` 中添加**:
```yaml
# Video subgoal 配置
use_video_subgoals: true
video_model_checkpoint: /path/to/video/model/checkpoint.pt
num_vid_pred_per_ep: 5
use_vid_first_n_frames: 2
video_future_horizon: 7
video_var_temp: 1.0
is_video_ddim: false
```

**在 `run_eval_ddp_causal.sh` 中添加**:
```bash
USE_VIDEO_SUBGOALS=${8:-false}
VIDEO_MODEL_CHECKPOINT=${9:-""}
```

### 8. 错误处理

**需要考虑的错误情况**:
- Video model 加载失败
- Video model 推理失败
- 图像格式转换错误
- GPU 内存不足

**建议的错误处理**:
```python
try:
    subgoals = self._generate_subgoals(current_obs, task_description)
except Exception as e:
    logger.warning(f"Failed to generate subgoals: {e}, falling back to original method")
    # 回退到原有逻辑
    subgoals = None
```

## 集成步骤总结

1. **准备 Video Model**:
   - 下载 video model checkpoint
   - 确认模型路径和配置

2. **修改 AR Policy**:
   - 在 `ar.py` 中添加 video model 初始化
   - 实现 subgoal 生成逻辑
   - 修改 `get_actions()` 集成 subgoal

3. **修改 Vidar Server** (如果需要):
   - 添加 subgoal frames 输入接口
   - 修改推理逻辑使用 subgoal

4. **更新配置**:
   - 在 `deploy_policy.yml` 中添加新参数
   - 在 `run_eval_ddp_causal.sh` 中添加参数传递

5. **测试**:
   - 单任务测试
   - 多任务分布式测试
   - 性能对比测试

## 潜在问题和解决方案

### 问题1: 图像分辨率不匹配
- **问题**: Video model 使用 128x128，Vidar 使用 640x736
- **解决**: 在转换时进行 resize，或修改 video model 输入分辨率

### 问题2: 坐标系不一致
- **问题**: 两个库可能使用不同的相机坐标系
- **解决**: 需要确认并统一坐标系，或进行坐标转换

### 问题3: 任务描述格式差异
- **问题**: 两个库的任务描述格式不同
- **解决**: 实现格式转换函数（见上文）

### 问题4: 分布式环境下的模型加载
- **问题**: 在 DDP 环境下，每个进程都需要加载 video model
- **解决**: 
  - 确保每个进程使用不同的 GPU
  - 考虑共享模型实例（如果可能）

## 性能影响评估

**预期影响**:
- **延迟增加**: 每次生成 subgoal 需要额外 0.5-2 秒（取决于模型大小）
- **内存增加**: Video model 需要额外 2-8GB GPU 内存
- **成功率**: 可能提升（如果 subgoal 有助于规划）

**优化建议**:
- 批量生成 subgoals
- 使用更小的模型或量化
- 异步生成 subgoals

## 测试建议

1. **单元测试**:
   - 测试图像格式转换
   - 测试 subgoal 生成
   - 测试错误处理

2. **集成测试**:
   - 测试完整的 pipeline
   - 测试与原有功能的兼容性

3. **性能测试**:
   - 对比有无 video subgoals 的性能
   - 测量延迟和内存占用

4. **消融实验**:
   - 测试不同的 subgoal 生成策略
   - 测试不同的参数设置

## 参考代码位置

- **Video Model 加载**: `video-to-action-release/diffuser/libero/lb_eval_utils.py`
- **Subgoal 生成逻辑**: `video-to-action-release/diffuser/libero/lb_eval_helper.py` (eval_1_env 方法)
- **AR Policy**: `vidar-robotwin/policy/AR/ar.py`
- **评估入口**: `vidar-robotwin/script/eval_policy.py`

## 下一步行动

1. 确定集成方案（推荐方案1或方案3）
2. 实现图像格式转换函数
3. 在 AR model 中添加 video model 初始化
4. 实现 subgoal 生成和集成逻辑
5. 更新配置文件和脚本
6. 进行测试和调试

