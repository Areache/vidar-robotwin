#!/usr/bin/env python
# -- coding: UTF-8
"""
独立的 subgoal 生成脚本，在 vm Python 环境中运行
通过 stdin/stdout 与主进程通信（JSON 格式）
"""
import sys
import json
import base64
import cv2
import numpy as np
import torch
import torchvision
from base64 import b64encode, b64decode

# 添加 video-to-action-release 的路径
video_action_path = '/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm'
if video_action_path not in sys.path:
    sys.path.insert(0, video_action_path)

# 延迟导入，在需要时才导入（避免启动时失败）
try:
    from diffuser.libero.lb_video_model_utils import lb_get_video_model_gcp_v2
    from diffuser.datasets.img_utils import imgs_preproc_simple_noCrop_v1
    _imports_ok = True
except ImportError as e:
    _imports_ok = False
    _import_error = str(e)


def base64_to_video_input(obs_b64):
    """将 base64 图像转换为 video model 输入格式"""
    # 解码
    img_bytes = b64decode(obs_b64)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 128x128 (video model 期望的尺寸)
    img_resized = cv2.resize(img_rgb, (128, 128))
    
    # Normalize and convert to tensor
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 128, 128)
    
    return img_tensor


def tensor_to_base64_vidar(img_tensor):
    """将 tensor 转换为 vidar 格式的 base64"""
    # 确保是 (1, 3, H, W) 格式
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    
    # 转换为 [0, 255] uint8
    img = (img_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # RGB -> BGR (vidar 使用 BGR)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Encode to JPEG
    img_tensor_bgr = torch.from_numpy(img_bgr).permute(2, 0, 1)
    jpeg_tensor = torchvision.io.encode_jpeg(img_tensor_bgr)
    img_b64 = b64encode(jpeg_tensor.numpy().tobytes()).decode('utf-8')
    
    return img_b64


def extract_task_from_prompt(prompt):
    """从 prompt 中提取任务描述"""
    if "performing the following task: " in prompt:
        return prompt.split("performing the following task: ")[-1].strip()
    return prompt


def load_video_model(model_path, milestone=24, device='cuda'):
    """加载 video model"""
    print(f"Loading LIBERO video model from {model_path}...", file=sys.stderr)
    print(f"  Milestone: {milestone}", file=sys.stderr)
    print(f"  Device: {device}", file=sys.stderr)
    sys.stderr.flush()
    
    print("  Calling lb_get_video_model_gcp_v2...", file=sys.stderr)
    sys.stderr.flush()
    video_model = lb_get_video_model_gcp_v2(
        ckpts_dir=model_path,
        milestone=milestone,
        flow=False
    )
    
    print("  Moving model to device...", file=sys.stderr)
    sys.stderr.flush()
    video_device = torch.device(device if torch.cuda.is_available() else 'cpu')
    video_model = video_model.to(video_device)
    video_model.eval()
    
    print("  Setting model parameters...", file=sys.stderr)
    sys.stderr.flush()
    # 设置模型参数
    video_model.ema.ema_model.var_temp = 1.0
    video_model.ema.ema_model.is_ddim_sampling = False
    
    print(f"✓ LIBERO video model loaded successfully on {video_device}", file=sys.stderr)
    sys.stderr.flush()
    return video_model, video_device


def generate_subgoals_with_interval(model, device, first_frame_obs_b64, instruction, num_subgoals, subgoal_interval=8):
    """生成间隔指定帧数的 subgoals"""
    # 1. 转换 base64 图像为 video model 输入格式
    img_tensor = base64_to_video_input(first_frame_obs_b64)
    img_tensor = img_tensor.to(device)
    
    # 2. 提取任务描述
    task_str = extract_task_from_prompt(instruction)
    tasks_str = [task_str]
    
    # 3. 生成 subgoals
    subgoals_b64 = []
    current_img_tensor = img_tensor
    
    # 生成第一个subgoal（第0帧，即第一帧本身）
    first_subgoal_b64 = first_frame_obs_b64
    subgoals_b64.append(first_subgoal_b64)
    
    # 生成后续的subgoals（每间隔指定帧数）
    for i in range(1, num_subgoals):
        # 使用video model生成下一个subgoal
        with torch.no_grad():
            preds_video = model.forward(current_img_tensor, tasks_str)
        
        # preds_video: List[tensor], 每个 tensor shape (1, T, 3, H, W)
        assert len(preds_video) == 1
        pred_v = preds_video[0]  # (T, 3, H, W), T = 7 (通常)
        
        # 选择对应帧的subgoal
        if pred_v.shape[0] >= subgoal_interval:
            subgoal_frame_idx = min(subgoal_interval - 1, pred_v.shape[0] - 1)
        else:
            subgoal_frame_idx = pred_v.shape[0] - 1
        
        subgoal_tensor = pred_v[subgoal_frame_idx].unsqueeze(0)  # (1, 3, H, W)
        subgoal_b64 = tensor_to_base64_vidar(subgoal_tensor)
        subgoals_b64.append(subgoal_b64)
        
        # 更新current_img_tensor为当前subgoal，用于生成下一个
        current_img_tensor = pred_v[subgoal_frame_idx].unsqueeze(0).to(device)
        current_img_tensor = torch.clamp(current_img_tensor, 0.0, 1.0)
    
    return subgoals_b64


# 全局变量存储模型（避免重复加载）
_video_model = None
_video_device = None

def main():
    """主函数：从 stdin 逐行读取 JSON 请求，处理并输出 JSON 结果"""
    global _video_model, _video_device
    
    # 输出启动信息到 stderr（不会干扰 JSON 输出）
    print("VM subgoal generator started, waiting for commands...", file=sys.stderr)
    sys.stderr.flush()
    
    # 检查导入是否成功
    if not _imports_ok:
        error_response = {
            'success': False,
            'error': f'Failed to import required modules: {_import_error}',
            'error_type': 'ImportError'
        }
        print(json.dumps(error_response))
        sys.stdout.flush()
        print(f"Import error: {_import_error}", file=sys.stderr)
        sys.stderr.flush()
        return
    
    # 循环处理多个请求
    for line in sys.stdin:
        try:
            # 解析 JSON 请求
            request = json.loads(line.strip())
            command = request.get('command')
            
            if command == 'load_model':
                # 加载模型
                model_path = request['model_path']
                milestone = request.get('milestone', 24)
                device = request.get('device', 'cuda')
                
                print(f"Received load_model command: path={model_path}, milestone={milestone}, device={device}", file=sys.stderr)
                sys.stderr.flush()
                
                _video_model, _video_device = load_video_model(model_path, milestone, device)
                
                print("Model loading completed, sending success response", file=sys.stderr)
                sys.stderr.flush()
                
                response = {
                    'success': True,
                    'message': 'Model loaded successfully'
                }
                
                # 输出 JSON 结果到 stdout（立即发送响应）
                response_json = json.dumps(response)
                print(response_json)
                sys.stdout.flush()
                print(f"Success response sent to stdout: {response_json[:100]}...", file=sys.stderr)
                sys.stderr.flush()
                # 确保数据真的被发送（fsync 可能在某些管道上失败，忽略错误）
                try:
                    import os
                    if hasattr(sys.stdout, 'fileno'):
                        fd = sys.stdout.fileno()
                        if fd >= 0:  # 有效的文件描述符
                            os.fsync(fd)
                except (OSError, IOError) as e:
                    # fsync 失败不影响功能，只是日志记录
                    print(f"Warning: fsync failed (non-critical): {e}", file=sys.stderr)
                continue  # 继续等待下一个命令
            
            elif command == 'generate_subgoals':
                # 生成 subgoals
                if _video_model is None:
                    # 如果模型未加载，先加载
                    model_path = request.get('model_path')
                    if model_path is None:
                        raise ValueError("model_path is required for first call")
                    milestone = request.get('milestone', 24)
                    device = request.get('device', 'cuda')
                    _video_model, _video_device = load_video_model(model_path, milestone, device)
                
                first_frame_obs_b64 = request['first_frame_obs_b64']
                instruction = request['instruction']
                num_subgoals = request.get('num_subgoals', 10)
                subgoal_interval = request.get('subgoal_interval', 8)
                
                subgoals = generate_subgoals_with_interval(
                    _video_model, _video_device,
                    first_frame_obs_b64, instruction,
                    num_subgoals, subgoal_interval
                )
                
                response = {
                    'success': True,
                    'subgoals': subgoals,
                    'num_subgoals': len(subgoals)
                }
                
            else:
                raise ValueError(f"Unknown command: {command}")
            
            # 输出 JSON 结果到 stdout（对于 generate_subgoals 命令）
            print(json.dumps(response))
            sys.stdout.flush()
            print(f"Response sent for command: {command}", file=sys.stderr)
            sys.stderr.flush()
            # 尝试 fsync（可能失败，但不影响功能）
            try:
                import os
                if hasattr(sys.stdout, 'fileno'):
                    fd = sys.stdout.fileno()
                    if fd >= 0:
                        os.fsync(fd)
            except (OSError, IOError):
                pass  # 忽略 fsync 错误（管道不支持 fsync）
            
        except json.JSONDecodeError as e:
            # JSON 解析错误
            error_response = {
                'success': False,
                'error': f'JSON decode error: {str(e)}',
                'error_type': 'JSONDecodeError'
            }
            print(json.dumps(error_response))
            sys.stdout.flush()
            
        except Exception as e:
            # 其他错误
            error_response = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
            print(json.dumps(error_response), file=sys.stderr)
            print(json.dumps(error_response))
            sys.stdout.flush()
            # 尝试 fsync（可能失败，但不影响功能）
            try:
                import os
                if hasattr(sys.stdout, 'fileno'):
                    fd = sys.stdout.fileno()
                    if fd >= 0:
                        os.fsync(fd)
            except (OSError, IOError):
                pass  # 忽略 fsync 错误


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        # 捕获所有未处理的异常（包括导入错误）
        error_response = {
            'success': False,
            'error': str(e),
            'error_type': type(e).__name__
        }
        print(json.dumps(error_response))
        sys.stdout.flush()
        # 尝试 fsync（可能失败，但不影响功能）
        try:
            import os
            if hasattr(sys.stdout, 'fileno'):
                fd = sys.stdout.fileno()
                if fd >= 0:
                    os.fsync(fd)
        except (OSError, IOError):
            pass  # 忽略 fsync 错误
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

