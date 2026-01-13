# Video Model Subgoal 集成使用指南

## 概述

本集成将 `video-to-action-release` 库的 video model 生成的 subgoal frames 集成到 `vidar-robotwin` 的 AR policy pipeline 中，作为 vidar 模型的条件输入。

## 已实现的功能

### 1. 图像格式转换工具 (`policy/AR/video_format_utils.py`)
- `convert_obs_to_video_input()`: 将vidar的base64图像转换为video model输入格式
- `convert_subgoal_to_vidar_format()`: 将video model输出转换为vidar可用的base64格式
- `extract_task_description()`: 从vidar的完整指令中提取简洁任务描述

### 2. AR类集成 (`policy/AR/ar.py`)
- 延迟加载video model（避免不必要的内存占用）
- Subgoal生成和队列管理
- 自动将subgoal frames传递给vidar server

### 3. Server接口修改 (`vidar/server/causal_worker.py`)
- 添加 `subgoal_frames` 参数支持
- 将subgoal frames合并到条件帧中

### 4. 配置和参数传递
- `deploy_policy.yml`: 添加video model配置参数
- `run_eval_ddp.py`: 添加命令行参数
- `run_eval_ddp_causal.sh`: 添加bash脚本参数

## 使用方法

### 基本使用（禁用video subgoals）

```bash
bash /mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar-robotwin/run_eval_ddp_causal.sh \
    hd_clean \
    /path/to/vidarc.pt \
    /path/to/idm.pt \
    ddp_causal \
    16 \
    10 \
    3.0
```

### 启用video subgoals

使用配置文件方式（类似`plan_lb.py`的方式），从配置文件中读取`vid_diffusion`参数：

```bash
bash /mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar-robotwin/run_eval_ddp_causal.sh \
    hd_clean \
    /path/to/vidarc.pt \
    /path/to/idm.pt \
    ddp_causal \
    16 \
    10 \
    3.0 \
    true \
    "config/libero/lb_tk8_65to72.py" \
    2 \
    5
```

### 参数说明

| 位置 | 参数名 | 默认值 | 说明 |
|------|--------|--------|------|
| $1 | TASK_CONFIG | hd_clean | 任务配置 |
| $2 | MODEL | vidarc.pt | Vidar模型路径 |
| $3 | IDM | idm.pt | IDM模型路径 |
| $4 | PREFIX | ddp_causal | 输出前缀 |
| $5 | NUM_NEW_FRAMES | 16 | 新帧数量 |
| $6 | NUM_SAMPLING_STEP | 10 | 采样步数 |
| $7 | CFG | 3.0 | 引导尺度 |
| $8 | USE_VIDEO_SUBGOALS | false | 是否启用video subgoals |
| $9 | VIDEO_MODEL_CONFIG_PATH | "" | 配置文件路径（如"config/libero/lb_tk8_65to72.py"） |
| $10 | USE_VID_FIRST_N_FRAMES | 2 | 使用前N帧作为subgoal |
| $11 | NUM_VID_PRED_PER_EP | 5 | 每个episode预测视频次数 |

**注意**：
- 配置文件需要包含`base['diffusion']['vid_diffusion']`字典，格式与`plan_lb.py`使用的配置文件相同

## 配置参数

在 `deploy_policy.yml` 中可以配置以下参数：

```yaml
# Video model subgoal configuration
use_video_subgoals: false  # 是否启用
video_model_config_path: "config/libero/lb_tk8_65to72.py"  # 配置文件路径
use_vid_first_n_frames: 2  # 使用前N帧
num_vid_pred_per_ep: 5  # 每个episode预测次数
```

### 配置文件格式

配置文件需要包含以下结构：

```python
base = {
    'diffusion': {
        'vid_diffusion': dict(
            ckpts_dir='./ckpts/libero/libero_ep20_bs12_aug',
            milestone=180000,
            timestep=100,
            g_w=0,
            cls_free_prob=0.0,
            sample_per_seq=8,
        ),
        # ... 其他配置
    },
    # ... 其他配置
}
```

这与`plan_lb.py`使用的配置文件格式完全一致。

## 工作流程

1. **初始化阶段**:
   - AR类初始化时，如果 `use_video_subgoals=true`，会准备video model相关配置
   - Video model采用延迟加载策略，只在首次需要时加载

2. **Subgoal生成**:
   - 当 `get_actions()` 被调用时，如果subgoal队列为空且未达到预测次数上限，会生成新的subgoals
   - 使用当前最新观察和任务描述调用video model
   - 提取前N帧作为subgoals并加入队列

3. **Subgoal使用**:
   - 每次调用 `get_actions()` 时，从队列中取出一个subgoal
   - 将subgoal作为额外条件帧传递给vidar server

4. **Server处理**:
   - Server接收subgoal frames，将其合并到条件帧中
   - Vidar模型使用包含subgoal的条件帧进行视频预测

## 注意事项

### 1. 依赖要求
- 确保 `video-to-action-release` 库已安装
- 确保两个库的PyTorch版本兼容
- 确保CUDA版本兼容

### 2. 环境变量
可能需要设置：
```bash
export PYTHONPATH=/path/to/video-to-action-release:$PYTHONPATH
```

### 3. 内存和性能
- Video model需要额外2-8GB GPU内存
- 每次生成subgoal需要0.5-2秒（取决于模型大小）
- 在DDP环境下，每个进程都需要独立加载video model

### 4. 错误处理
- 如果video model加载失败，会自动回退到原有逻辑（不使用subgoals）
- 如果subgoal生成失败，会返回空列表，不影响原有流程

### 5. 图像格式
- Video model使用128x128 RGB格式
- Vidar使用640x736 BGR格式
- 转换函数会自动处理格式转换和resize

## 故障排查

### Video model加载失败
- 检查checkpoint路径是否正确
- 检查milestone编号是否存在
- 检查CUDA设备是否可用
- 查看日志中的错误信息

### Subgoal生成失败
- 检查图像格式转换是否正确
- 检查任务描述提取是否正确
- 查看AR类的日志输出

### Server接收subgoal失败
- 检查subgoal_frames参数是否正确传递
- 检查base64编码是否正确
- 查看server日志

## 测试建议

1. **单任务测试**（不使用DDP）:
   ```bash
   # 在eval_policy.py中直接测试
   ```

2. **多任务分布式测试**:
   ```bash
   # 使用run_eval_ddp_causal.sh进行完整测试
   ```

3. **性能对比**:
   - 对比有无video subgoals的成功率
   - 测量延迟和内存占用
   - 记录subgoal生成时间

## 下一步优化

1. **性能优化**:
   - 异步生成subgoals
   - 批量生成subgoals
   - 模型量化

2. **功能增强**:
   - 支持动态调整subgoal生成频率
   - 支持subgoal质量评估
   - 支持subgoal缓存和重用

3. **代码优化**:
   - 更好的错误处理和日志
   - 更灵活的配置选项
   - 更完善的单元测试

