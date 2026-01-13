# 交互式调试指南

由于代码在子进程中执行，传统的 `pdb.set_trace()` 无法直接使用。以下是几种实用的交互调试方案。

## 方案1: 远程调试器 (推荐) ⭐

### 安装依赖
```bash
pip install remote-pdb
# 或
pip install rpdb
```

### 使用方法

#### 在代码中添加调试点
```python
# 在 ar.py 的 get_actions() 方法中
from policy.AR.debug_utils import setup_remote_pdb2

def get_actions(self):
    # ... 你的代码 ...
    
    # 添加调试点
    setup_remote_pdb2(port=4444)  # 可以设置不同端口避免冲突
    
    # ... 继续执行 ...
```

#### 连接调试器

**方式A: 使用 telnet**
```bash
# 在另一个终端
telnet localhost 4444
# 或如果在不同机器
telnet <server_ip> 4444
```

**方式B: 使用 nc (netcat)**
```bash
nc localhost 4444
```

**方式C: 使用 Python**
```python
import socket
s = socket.socket()
s.connect(('localhost', 4444))
# 现在可以发送调试命令
```

### 调试命令
连接后，可以使用标准的 pdb 命令：
- `n` (next): 下一行
- `s` (step): 进入函数
- `c` (continue): 继续执行
- `p <variable>`: 打印变量
- `pp <variable>`: 美化打印
- `l` (list): 显示代码
- `w` (where): 显示调用栈
- `q` (quit): 退出调试

## 方案2: 环境变量控制调试模式

### 修改 run_eval_ddp.py
已经修改为支持 `DEBUG_MODE` 环境变量。

### 使用方法
```bash
# 启用调试模式（输出到终端，允许交互）
export DEBUG_MODE=1
bash run_eval_ddp_causal.sh hd_clean "" "" "" "" "" "" true "config/libero/lb_tk8_65to72.py"

# 正常模式（输出到日志文件）
unset DEBUG_MODE
bash run_eval_ddp_causal.sh ...
```

## 方案3: 条件断点

### 使用场景
只在特定条件下触发调试，比如：
- 特定任务
- 特定变量值
- 特定错误

### 代码示例
```python
from policy.AR.debug_utils import conditional_breakpoint

def get_actions(self):
    # 只在启用 video subgoals 时调试
    conditional_breakpoint(
        condition=self.use_video_subgoals and len(self.obs_cache) > 0,
        port=4444
    )
```

## 方案4: 文件触发调试

### 使用方法
```python
from policy.AR.debug_utils import debug_if_file_exists

def get_actions(self):
    # 检查触发文件
    debug_if_file_exists(trigger_file="/tmp/debug_trigger", port=4444)
```

### 触发调试
```bash
# 需要调试时创建文件
touch /tmp/debug_trigger

# 调试完成后删除
rm /tmp/debug_trigger
```

## 方案5: 增强日志调试

### 使用 DebugLogger
```python
from policy.AR.debug_utils import DebugLogger

# 在类初始化时
self.debug_logger = DebugLogger(
    log_file="/tmp/debug.log",
    enable_interactive=False  # 设为 True 启用远程调试
)

def get_actions(self):
    self.debug_logger.log("get_actions called")
    self.debug_logger.inspect(self.obs_cache, "obs_cache")
    self.debug_logger.inspect(self.use_video_subgoals, "use_video_subgoals")
    
    # 需要断点时
    self.debug_logger.breakpoint("Before subgoal generation", port=4444)
```

## 方案6: 直接运行（不使用 subprocess）

### 临时修改代码
如果想直接调试，可以临时修改 `run_eval_ddp.py`：

```python
# 注释掉 subprocess.run，直接导入执行
# with open(log_file, "w") as f:
#     subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False, env=env)

# 直接执行
import sys
sys.path.insert(0, os.getcwd())
from script.eval_policy import main as eval_main
# 手动设置参数并调用
```

### 或者直接运行
```bash
cd /mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar-robotwin
python script/eval_policy.py \
    --config policy/AR/deploy_policy.yml \
    --overrides \
    --task_name click_bell \
    --task_config hd_clean \
    --port 25400 \
    --seed 1234 \
    --policy_name AR \
    --num_new_frames 16 \
    --num_sampling_step 10 \
    --guide_scale 3.0 \
    --rollout_bound 60 \
    --rollout_prefill_num 33 \
    --save_dir /tmp/test_debug
```

然后在代码中使用标准 `pdb`：
```python
import pdb; pdb.set_trace()
```

## 方案7: IDE 远程调试

### VS Code
1. 安装 Python Debugger 扩展
2. 创建 `.vscode/launch.json`:
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Remote Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            }
        }
    ]
}
```

3. 在代码中：
```python
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()  # 可选：等待连接
```

### PyCharm
1. 创建 Remote Debug 配置
2. 在代码中添加：
```python
import pydevd_pycharm
pydevd_pycharm.settrace('localhost', port=12345, stdoutToServer=True, stderrToServer=True)
```

## 推荐工作流

### 快速调试流程
1. **第一步**: 使用增强日志 (`DebugLogger`) 快速定位问题
2. **第二步**: 如果需要在特定点交互，使用远程调试器 (`setup_remote_pdb2`)
3. **第三步**: 对于复杂问题，使用 IDE 远程调试

### 示例：调试 video subgoals
```python
from policy.AR.debug_utils import DebugLogger, setup_remote_pdb2

def get_actions(self):
    # 创建调试器
    debugger = DebugLogger(enable_interactive=True)
    
    debugger.log("=== get_actions called ===")
    debugger.inspect(self.use_video_subgoals, "use_video_subgoals")
    debugger.inspect(len(self.obs_cache), "obs_cache length")
    
    if self.use_video_subgoals:
        # 只在需要时触发远程调试
        if len(self.obs_cache) > 0:
            debugger.breakpoint("Before generating subgoals", port=4444)
            # 或直接
            # setup_remote_pdb2(port=4444)
```

## 注意事项

1. **端口冲突**: 在分布式环境中，每个进程使用不同端口
   ```python
   import os
   port = 4444 + int(os.environ.get("LOCAL_RANK", 0))
   setup_remote_pdb2(port=port)
   ```

2. **防火墙**: 确保调试端口可访问

3. **性能影响**: 调试会暂停程序执行，注意超时问题

4. **生产环境**: 确保调试代码不会进入生产环境

## 快速参考

| 方案 | 适用场景 | 复杂度 | 推荐度 |
|------|----------|--------|--------|
| 远程调试器 | 需要交互式调试 | 低 | ⭐⭐⭐⭐⭐ |
| 环境变量控制 | 临时启用调试 | 低 | ⭐⭐⭐⭐ |
| 条件断点 | 特定条件触发 | 中 | ⭐⭐⭐⭐ |
| 文件触发 | 动态控制调试 | 中 | ⭐⭐⭐ |
| 增强日志 | 快速定位问题 | 低 | ⭐⭐⭐⭐⭐ |
| 直接运行 | 简单场景 | 低 | ⭐⭐⭐ |
| IDE 远程调试 | 复杂调试 | 高 | ⭐⭐⭐⭐ |










