# -- coding: UTF-8
import numpy as np
import json
import requests
import cv2
import urllib3
from base64 import b64encode, b64decode
import os
import multiprocessing
import subprocess
import logging
import torch
import torchvision
import time
from datetime import datetime
import sys

# VM Python 环境路径（用于运行 subgoal 生成）
python_vm = "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/env/v4a/bin/python"
vm_subgoal_script = os.path.join(os.path.dirname(__file__), "vm_subgoal_generator.py")


def save_video(ffmpeg_cmd, images):
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    for image in images:
        img = cv2.imdecode(np.frombuffer(b64decode(image), np.uint8), cv2.IMREAD_COLOR)
        proc.stdin.write(img.tobytes())
    proc.stdin.close()
    proc.wait()


def save_videos(videos, width, height, fps=8):
    workers = []
    for k, v in videos.items():
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', '-c:v', 'libx264', '-preset', 'veryslow',
            '-crf', '10', '-threads', '1', '-pix_fmt', 'yuv420p',
            '-loglevel', 'error', k
        ]
        workers.append(multiprocessing.Process(target=save_video, args=(ffmpeg_cmd, v)))
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


def worker(port, headers, data, verify):
    response = requests.post(f"http://localhost:{port}", headers=headers, data=json.dumps(data), verify=verify).json()
    assert len(response) > 0, "password error"
    return response


class AR:
    def __init__(self, usr_args=None, version=None):
        if usr_args is None:
            usr_args = {}
        
        # Apply version parameters if version is specified
        if version:
            try:
                from .version_config import load_version_config
                version_config = load_version_config()
                usr_args = version_config.apply_to_usr_args(usr_args, version)
                print(f"Applied version parameters for: {version}")
                print(f"DEBUG [AR.__init__]: After version apply - use_mpc={usr_args.get('use_mpc')}, "
                      f"mpc_num_candidates={usr_args.get('mpc_num_candidates')}, "
                      f"mpc_cost_weights={usr_args.get('mpc_cost_weights')}")
            except Exception as e:
                print(f"Warning: Failed to load version {version}: {e}. Using default parameters.")
        
        self.usr_args = usr_args
        self.policy_name = usr_args["policy_name"]
        self.task_name = usr_args["task_name"]
        self.task_config = usr_args["task_config"]
        self.num_new_frames = usr_args["num_new_frames"]
        self.num_sampling_step = usr_args["num_sampling_step"]
        self.max_steps = usr_args["max_steps"]
        self.seed = usr_args["seed"]
        self.port = usr_args["port"]  # Vidar 服务器端口
        self.save_dir = usr_args["save_dir"]
        self.obs_cache = []
        self.prompt = None
        self.episode_id = 0
        self.timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.out_imgs = []
        self.out_masks = []
        self.video_ffmpeg = None
        self.num_conditional_frames = None
        self.rollout_bound = usr_args["rollout_bound"]
        self.rollout_prefill_num = usr_args["rollout_prefill_num"]
        self.guide_scale = usr_args["guide_scale"]
        
        # MPC 配置（v2 版本）
        use_mpc_raw = usr_args.get("use_mpc", False)
        if isinstance(use_mpc_raw, str):
            self.use_mpc = use_mpc_raw.lower() in ("true", "1", "yes")
        else:
            self.use_mpc = bool(use_mpc_raw)
        self.mpc_num_candidates = usr_args.get("mpc_num_candidates", 50)
        self.mpc_cost_weights = usr_args.get("mpc_cost_weights", {'task': 1.0, 'ctrl': 0.1, 'reach': 0.5})
        print(f"DEBUG [AR.__init__]: MPC config initialized - use_mpc={self.use_mpc}, "
              f"mpc_num_candidates={self.mpc_num_candidates}, "
              f"mpc_cost_weights={self.mpc_cost_weights}")
        
        # Diffusion forcing 配置（df 版本）
        use_diffusion_forcing_raw = usr_args.get("use_diffusion_forcing", False)
        if isinstance(use_diffusion_forcing_raw, str):
            self.use_diffusion_forcing = use_diffusion_forcing_raw.lower() in ("true", "1", "yes")
        else:
            self.use_diffusion_forcing = bool(use_diffusion_forcing_raw)
        
        # LIBERO subgoal 配置（支持直接调用 model 或 HTTP 服务器）
        # 兼容字符串和布尔值类型
        use_subgoal_raw = usr_args.get("use_libero_subgoal", False)
        if isinstance(use_subgoal_raw, str):
            self.use_libero_subgoal = use_subgoal_raw.lower() in ("true", "1", "yes")
        else:
            self.use_libero_subgoal = bool(use_subgoal_raw)
        
        # libero_use_direct_model 也兼容字符串和布尔值
        direct_model_raw = usr_args.get("libero_use_direct_model", True)
        if isinstance(direct_model_raw, str):
            self.libero_use_direct_model = direct_model_raw.lower() in ("true", "1", "yes")
        else:
            self.libero_use_direct_model = bool(direct_model_raw)
        self.libero_subgoal_port = usr_args.get("libero_subgoal_port", 25401)
        self.libero_subgoal_url = f"http://localhost:{self.libero_subgoal_port}"
        self.current_subgoals = []  # 当前可用的 subgoals
        self.subgoal_idx = 0  # 当前使用的 subgoal 索引
        self.first_frame_obs = None  # 第一帧观察（用于生成subgoals）
        self.subgoal_interval = 8  # subgoal间隔帧数
        
        # Video model 直接调用配置
        self.video_model = None
        self.video_device = None
        self.rendered_imgs_preproc_fn = None
        self.libero_model_path = usr_args.get("libero_model_path", None)
        self.libero_logbase = usr_args.get("libero_logbase", "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm/logs")
        self.libero_dataset = usr_args.get("libero_dataset", "libero-8tk-65to72-v3")
        self.libero_exp_name = usr_args.get("libero_exp_name", "lb_tk8_65to72")
        self.libero_epoch = usr_args.get("libero_epoch", "latest")
        self.libero_milestone = usr_args.get("libero_milestone", 24)
        
        # VM subprocess 进程（用于在独立的 Python 环境中运行）
        self.vm_process = None
        
        # 如果启用直接调用，初始化 video model（通过 subprocess）
        if self.use_libero_subgoal and self.libero_use_direct_model:
            self._init_video_model()
        
        os.makedirs(self.save_dir, exist_ok=True)

    def set_ffmpeg(self, save_path):
        """
        设置 ffmpeg 进程用于保存视频
        
        Args:
            save_path: 保存视频的路径
            
        Raises:
            ValueError: 如果路径无效
            FileNotFoundError: 如果 ffmpeg 不存在
        """
        # 验证保存路径
        if not save_path:
            raise ValueError("save_path cannot be empty")
        
        # 确保目录存在
        save_dir = os.path.dirname(save_path)
        if save_dir:
            try:
                os.makedirs(save_dir, exist_ok=True)
            except (OSError, PermissionError) as e:
                raise ValueError(f"Cannot create directory {save_dir}: {e}")
        
        # 验证路径是否可写
        try:
            # 尝试创建临时文件来测试写权限
            test_file = save_path + ".test"
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
        except (OSError, PermissionError) as e:
            raise ValueError(f"Cannot write to {save_path}: {e}")
        
        # 检查 ffmpeg 是否存在
        import shutil
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise FileNotFoundError(
                "ffmpeg not found in PATH. "
                "Please install ffmpeg or add it to your PATH."
            )
        
        # 验证路径长度（某些系统有路径长度限制）
        if len(save_path) > 255:
            print(f"Warning: save_path is very long ({len(save_path)} chars), may cause issues on some systems")
        
        try:
            self.video_ffmpeg = subprocess.Popen(
                    [
                        ffmpeg_path,
                        "-y",
                        "-loglevel",
                        "error",
                        "-f",
                        "rawvideo",
                        "-pixel_format",
                        "rgb24",
                        "-video_size",
                        "640x736",
                        "-framerate",
                        "10",
                        "-i",
                        "-",
                        "-pix_fmt",
                        "yuv420p",
                        "-vcodec",
                        "libx264",
                        "-crf",
                        "23",
                        save_path,  # 使用原始路径，不进行 f-string 格式化
                    ],
                    stdin=subprocess.PIPE,
                    stderr=subprocess.PIPE,  # 捕获 stderr 以便诊断
                )
        except Exception as e:
            raise RuntimeError(f"Failed to start ffmpeg process: {e}")
        
        # 验证进程是否成功启动
        if self.video_ffmpeg.poll() is not None:
            # 进程立即退出，读取错误信息
            stderr_output = self.video_ffmpeg.stderr.read().decode('utf-8', errors='ignore')
            raise RuntimeError(
                f"FFmpeg process exited immediately. "
                f"Error: {stderr_output or 'Unknown error'}. "
                f"Path: {save_path}"
            )

    def pause_ffmpeg(self):
        """暂停 ffmpeg 进程（用于调试，避免错误）"""
        if self.video_ffmpeg and self.video_ffmpeg.poll() is None:
            try:
                import signal
                self.video_ffmpeg.send_signal(signal.SIGSTOP)
                return True
            except (ProcessLookupError, OSError):
                return False
        return False
    
    def resume_ffmpeg(self):
        """恢复 ffmpeg 进程"""
        if self.video_ffmpeg and self.video_ffmpeg.poll() is None:
            try:
                import signal
                self.video_ffmpeg.send_signal(signal.SIGCONT)
                return True
            except (ProcessLookupError, OSError):
                return False
        return False
    
    def close_ffmpeg(self, timeout=10):
        """
        安全地关闭 ffmpeg 进程
        
        Args:
            timeout: 等待进程结束的超时时间（秒），默认 10 秒，调试时可以设置更长
        """
        if self.video_ffmpeg:
            try:
                # 检查进程是否还在运行
                if self.video_ffmpeg.poll() is None:
                    # 如果进程被暂停，先恢复它
                    try:
                        import signal
                        self.video_ffmpeg.send_signal(signal.SIGCONT)
                    except (ProcessLookupError, OSError):
                        pass
                    
                    # 进程还在运行，先关闭 stdin
                    try:
                        self.video_ffmpeg.stdin.close()
                    except (BrokenPipeError, OSError, ValueError):
                        # stdin 可能已经关闭，忽略错误
                        pass
                    
                    # 等待进程结束（使用可配置的超时时间）
                    try:
                        return_code = self.video_ffmpeg.wait(timeout=timeout)
                        if return_code != 0:
                            # 读取 stderr 以获取错误信息
                            try:
                                stderr_output = self.video_ffmpeg.stderr.read().decode('utf-8', errors='ignore')
                                if stderr_output:
                                    print(f"FFmpeg exited with code {return_code}. Error: {stderr_output}")
                            except Exception:
                                pass
                    except subprocess.TimeoutExpired:
                        # 如果超时，先尝试优雅终止
                        print(f"FFmpeg process did not finish within {timeout}s, terminating...")
                        self.video_ffmpeg.terminate()
                        try:
                            self.video_ffmpeg.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            # 最后强制杀死
                            print("Force killing ffmpeg process...")
                            self.video_ffmpeg.kill()
                            self.video_ffmpeg.wait()
                else:
                    # 进程已经退出，检查返回码
                    return_code = self.video_ffmpeg.returncode
                    if return_code != 0:
                        # 读取 stderr 以获取错误信息
                        try:
                            stderr_output = self.video_ffmpeg.stderr.read().decode('utf-8', errors='ignore')
                            if stderr_output:
                                print(f"FFmpeg exited with code {return_code}. Error: {stderr_output}")
                        except Exception:
                            pass
            except Exception as e:
                # 记录错误但继续清理
                print(f"Error closing ffmpeg: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # 确保清理 stderr
                try:
                    if self.video_ffmpeg.stderr:
                        self.video_ffmpeg.stderr.close()
                except Exception:
                    pass
                self.video_ffmpeg = None

    def reset(self):
        """Resets the internal state of the model."""
        self.obs_cache = []
        self.prompt = ""
        self.out_imgs = []
        self.out_masks = []
        self.num_conditional_frames = 1
        self.current_subgoals = []
        self.subgoal_idx = 0
        self.first_frame_obs = None
    
    def close_all_resources(self):
        """关闭所有资源（ffmpeg、VM subprocess 等），用于调试前清理"""
        self.close_ffmpeg()
        self._close_vm_process()
    
    def __del__(self):
        """清理资源，关闭 VM subprocess"""
        self._close_vm_process()


    def update_obs(self, obs):
        """Updates the model with the latest observation."""
        img = torch.tensor(obs, dtype=torch.uint8).permute(2, 0, 1)
        img = torchvision.io.encode_jpeg(img)
        img = b64encode(img.numpy().tobytes()).decode("utf-8")
        self.obs_cache.append(img)

    def set_instruction(self, instruction):
        """Sets the task instruction for the policy."""

        system_prompt = "The whole scene is in a realistic, industrial art style with three views: a fixed rear camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: "
        if instruction:
            instruction = instruction[0].lower() + instruction[1:]
        self.prompt = system_prompt + instruction

    def set_demo_instruction(self, instruction):
        """Sets the task instruction for the policy."""

        system_prompt = "The whole scene is in a realistic, industrial art style with three views: a fixed front camera, a movable left arm camera, and a movable right arm camera. The aloha robot is currently performing the following task: "
        self.prompt = system_prompt + instruction


    def set_episode_id(self, episode_id):
        """Sets the episode ID for the current run."""
        self.episode_id = episode_id

    def modify_actions(self, actions):
        """
        以第一帧为基准，对夹爪角度的变化量做线性变换后施加回去，并将夹角限制在[0, 5]区间
        :param actions: shape (N, 14)
        :param gripper_strengthen_factor: 收紧趋势加强系数
        :param bias: 偏置
        :return: 修改后的actions
        """
        actions = np.array(actions)
        for dim in [6, 13]:
            smoothed = actions[:, dim].copy()
            for i in range(2, len(smoothed)-2):
                smoothed[i] = (actions[i-2, dim] + actions[i-1, dim] + actions[i, dim] + actions[i+1, dim] + actions[i+2, dim]) / 5
            actions[:, dim] = smoothed
            # if action is decreasing and action < 3, then set to 0
            diffs = actions[1:, dim] - actions[:-1, dim]
            mask = diffs < 0.1
            # append one more element to mask
            mask = np.concatenate(([False], mask))
            actions[:, dim] = np.where(mask & (actions[:, dim] < 0.3), np.clip(actions[:, dim],None,0), actions[:, dim])
            actions[:, dim] = np.where(actions[:, dim] > 0.7, np.clip(actions[:, dim], 1, None), actions[:, dim])
        return actions.tolist()

    def _init_video_model(self):
        """初始化 video model（通过 subprocess 在独立的 Python 环境中运行）"""
        try:
            print("Initializing LIBERO video model (subprocess mode)...")
            print(f"  Model path: {self.libero_model_path}")
            print(f"  Milestone: {self.libero_milestone}")
            print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
            
            # 检查模型路径
            if self.libero_model_path is None:
                print("Warning: libero_model_path not provided, trying to use default path")
                # 尝试使用默认路径
                default_path = "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm/ckpts/libero/libero_ep20_bs12_aug"
                if os.path.exists(default_path):
                    self.libero_model_path = default_path
                else:
                    print(f"Error: Default model path not found: {default_path}")
                    return
            
            # 检查 VM Python 环境和脚本是否存在
            print(f"  Checking VM Python: {python_vm}")
            if not os.path.exists(python_vm):
                print(f"Error: VM Python environment not found: {python_vm}")
                return
            
            print(f"  Checking VM script: {vm_subgoal_script}")
            if not os.path.exists(vm_subgoal_script):
                print(f"Error: VM subgoal script not found: {vm_subgoal_script}")
                return
            
            # 启动 subprocess（保持运行，通过 stdin/stdout 通信）
            print("  Starting VM subprocess...")
            # 使用 unbuffered 模式确保输出立即刷新
            # 注意：text=True 时 readline() 返回字符串，但可能返回 None（EOF）
            self.vm_process = subprocess.Popen(
                [python_vm, "-u", vm_subgoal_script],  # -u 参数启用 unbuffered 模式
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,  # 文本模式，readline() 返回字符串
                bufsize=1,  # 行缓冲（text mode 推荐使用 1）
                universal_newlines=True  # 确保跨平台兼容性
            )
            
            # 等待一小段时间确保 subprocess 启动
            import time
            time.sleep(0.5)
            
            # 检查 subprocess 是否还在运行
            if self.vm_process.poll() is not None:
                # subprocess 已经退出
                try:
                    stderr_output = self.vm_process.stderr.read()
                except Exception:
                    stderr_output = ""
                print(f"Error: VM subprocess exited immediately with code {self.vm_process.returncode}")
                if stderr_output:
                    print(f"Stderr output:\n{stderr_output}")
                self._close_vm_process()
                self.video_model = None
                return
            
            print("  VM subprocess started successfully")
            
            # 发送加载模型的命令
            print("  Sending load_model command...")
            load_request = {
                'command': 'load_model',
                'model_path': self.libero_model_path,
                'milestone': self.libero_milestone,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            
            request_json = json.dumps(load_request) + '\n'
            self.vm_process.stdin.write(request_json)
            self.vm_process.stdin.flush()
            
            # 读取响应（设置超时，模型加载可能需要较长时间）
            import threading
            
            stderr_content = []
            stderr_lock = threading.Lock()
            stderr_stop = threading.Event()
            
            # 使用线程异步读取 stderr，并实时打印
            def read_stderr():
                try:
                    while not stderr_stop.is_set():
                        line = self.vm_process.stderr.readline()
                        if not line:
                            if self.vm_process.poll() is not None:
                                break
                            time.sleep(0.1)
                            continue
                        with stderr_lock:
                            stderr_content.append(line)
                        # 实时打印 stderr（模型加载进度）
                        print(f"  [VM] {line.rstrip()}")
                except Exception as e:
                    print(f"  [VM stderr reader error] {e}")
            
            stderr_thread = threading.Thread(target=read_stderr, daemon=True)
            stderr_thread.start()
            
            # 读取响应（增加超时时间，模型加载可能需要几分钟）
            print("  Waiting for model to load (this may take several minutes)...")
            print("  (You should see loading progress messages above)")
            # 模型加载可能需要 2-5 分钟，设置 5 分钟超时
            print("  Reading response from VM subprocess...")
            # 等待一小段时间，确保响应有时间发送
            time.sleep(0.5)
            
            # 先尝试直接读取（可能响应已经在缓冲区中）
            print("  Attempting to read response...")
            try:
                # 尝试非阻塞读取一次
                import select
                if select.select([self.vm_process.stdout], [], [], 0.1)[0]:
                    line = self.vm_process.stdout.readline()
                    if line and line.strip():
                        try:
                            response = json.loads(line.strip())
                            print(f"  [DEBUG] Got response immediately: {response}")
                            print(f"  Response received: True")
                            if response:
                                print(f"  Response content: success={response.get('success')}, message={response.get('message', 'N/A')}")
                        except json.JSONDecodeError as e:
                            print(f"  [DEBUG] Not JSON: {line[:100]}, error: {e}")
                            response = None
                    else:
                        response = None
                else:
                    response = None
            except Exception as e:
                print(f"  [DEBUG] Direct read failed: {e}")
                response = None
            
            # 如果没有立即读取到，使用正常流程
            if response is None:
                print("  No immediate response, using normal read flow...")
                response = self._read_json_response_from_vm(timeout=300)
                print(f"  Response received: {response is not None}")
                if response:
                    print(f"  Response content: success={response.get('success')}, message={response.get('message', 'N/A')}")
            
            # 停止 stderr 读取线程
            stderr_stop.set()
            time.sleep(0.1)  # 给线程一点时间退出
            
            if response is not None:
                if response.get('success'):
                    print(f"✓ LIBERO video model loaded successfully (subprocess mode)")
                    self.video_model = True  # 标记为已加载
                else:
                    error_msg = response.get('error', 'Unknown error')
                    print(f"✗ Failed to load LIBERO video model: {error_msg}")
                    with stderr_lock:
                        stderr_output = ''.join(stderr_content)
                    if stderr_output:
                        print(f"Stderr output:\n{stderr_output}")
                    self._close_vm_process()
                    self.video_model = None
            else:
                print("✗ Failed to get response from VM subprocess (timeout after 5 minutes)")
                # 检查 subprocess 是否还在运行
                if self.vm_process.poll() is not None:
                    print(f"  VM subprocess exited with code {self.vm_process.returncode}")
                else:
                    print("  VM subprocess is still running (may be loading model)")
                with stderr_lock:
                    stderr_output = ''.join(stderr_content)
                if stderr_output:
                    print(f"Stderr output (last 50 lines):\n{''.join(stderr_content[-50:])}")
                else:
                    print("  No stderr output captured (subprocess may be stuck)")
                print("  Suggestion: Check if model loading is in progress, or increase timeout")
                self._close_vm_process()
                self.video_model = None
            
        except Exception as e:
            print(f"✗ Failed to initialize LIBERO video model: {e}")
            import traceback
            traceback.print_exc()
            self._close_vm_process()
            self.video_model = None
    
    def _close_vm_process(self):
        """关闭 VM subprocess"""
        if self.vm_process is not None:
            try:
                self.vm_process.stdin.close()
                self.vm_process.stdout.close()
                self.vm_process.stderr.close()
                self.vm_process.terminate()
                self.vm_process.wait(timeout=5)
            except Exception as e:
                print(f"Error closing VM process: {e}")
            finally:
                self.vm_process = None
    
    def _read_json_response_from_vm(self, timeout=60):
        """
        从 VM subprocess 读取 JSON 响应，跳过非 JSON 行（调试输出）
        同时监控 stderr 以捕获错误信息
        
        Returns:
            dict: JSON 响应，如果失败则返回 None
        """
        if self.vm_process is None:
            return None
        
        max_wait = timeout
        wait_time = 0
        
        # 使用 select 或非阻塞读取以提高响应速度
        import select
        import fcntl
        
        # 存储 stderr 输出（用于诊断）
        stderr_lines = []
        
        # 设置 stdout 为非阻塞模式（如果可能）
        try:
            # 获取当前标志
            flags = fcntl.fcntl(self.vm_process.stdout, fcntl.F_GETFL)
            # 添加非阻塞标志
            fcntl.fcntl(self.vm_process.stdout, fcntl.F_SETFL, flags | os.O_NONBLOCK)
            use_nonblocking = True
        except (OSError, AttributeError, ImportError):
            # 如果不支持非阻塞模式，使用原来的方式
            use_nonblocking = False
        
        try:
            # 首先尝试直接读取（可能响应已经在缓冲区中）
            print(f"  [DEBUG] Starting to read response, use_nonblocking={use_nonblocking}")
            if use_nonblocking:
                try:
                    # 尝试立即读取一次（非阻塞）
                    flags_backup = fcntl.fcntl(self.vm_process.stdout, fcntl.F_GETFL)
                    fcntl.fcntl(self.vm_process.stdout, fcntl.F_SETFL, flags_backup | os.O_NONBLOCK)
                    try:
                        line = self.vm_process.stdout.readline()
                        print(f"  [DEBUG] Immediate read result: {repr(line[:100]) if line else 'None'}")
                        if line and line.strip():
                            line = line.strip()
                            try:
                                response = json.loads(line)
                                print(f"  [DEBUG] Received response immediately: {line[:100]}...")
                                fcntl.fcntl(self.vm_process.stdout, fcntl.F_SETFL, flags_backup)
                                return response
                            except json.JSONDecodeError as e:
                                print(f"  [DEBUG] JSON decode error: {e}, line: {line[:100]}")
                                pass  # 不是 JSON，继续正常流程
                    finally:
                        fcntl.fcntl(self.vm_process.stdout, fcntl.F_SETFL, flags_backup)
                except Exception as e:
                    print(f"  [DEBUG] Immediate read exception: {e}")
                    pass  # 如果失败，继续正常流程
            
            print(f"  [DEBUG] Entering read loop, max_wait={max_wait}")
            iteration = 0
            while wait_time < max_wait:
                iteration += 1
                if iteration % 100 == 0:  # 每10秒打印一次进度
                    print(f"  [DEBUG] Still waiting for response... (waited {wait_time:.1f}s, iteration {iteration})")
                
                if self.vm_process.poll() is not None:
                    # subprocess 已退出，尝试读取最后的 stdout 和 stderr
                    print(f"  [DEBUG] Subprocess exited, reading remaining data...")
                    try:
                        # 尝试读取剩余的 stdout
                        remaining_stdout = self.vm_process.stdout.read()
                        print(f"  [DEBUG] Remaining stdout: {repr(remaining_stdout[:200]) if remaining_stdout else 'None'}")
                        if remaining_stdout:
                            lines = remaining_stdout.strip().split('\n')
                            for line in lines:
                                if line.strip():
                                    try:
                                        response = json.loads(line.strip())
                                        print(f"  [DEBUG] Received response after exit: {line[:100]}...")
                                        return response
                                    except json.JSONDecodeError as e:
                                        print(f"  [DEBUG] JSON decode error in remaining stdout: {e}")
                                        pass
                    except Exception as e:
                        print(f"  [DEBUG] Error reading remaining stdout: {e}")
                        pass
                    try:
                        remaining_stderr = self.vm_process.stderr.read()
                        if remaining_stderr:
                            stderr_lines.append(remaining_stderr)
                    except Exception:
                        pass
                    if stderr_lines:
                        print(f"VM subprocess stderr (before exit): {''.join(stderr_lines[-5:])}")
                    return None
                
                # 同时检查 stdout 和 stderr
                read_list = [self.vm_process.stdout]
                if self.vm_process.stderr:
                    read_list.append(self.vm_process.stderr)
                
                # 尝试读取一行
                if use_nonblocking:
                    try:
                        # 使用 select 检查是否有数据可读
                        ready, _, _ = select.select(read_list, [], [], 0.1)  # 增加等待时间到 0.1 秒
                        if self.vm_process.stdout in ready:
                            try:
                                # 检查 stdout 是否仍然有效
                                if self.vm_process.stdout is None:
                                    line = None
                                else:
                                    line = self.vm_process.stdout.readline()
                                    # readline() 可能返回 None（EOF）或空字符串
                                    if line is not None and line:
                                        # 确保 line 是字符串类型
                                        if isinstance(line, bytes):
                                            line = line.decode('utf-8', errors='ignore')
                                        # 调试：打印接收到的原始行（前100字符）
                                        line_preview = line.strip()[:100]
                                        if line_preview:
                                            print(f"  [DEBUG] Received from stdout: {line_preview}...")
                                    else:
                                        line = None
                            except (OSError, IOError, TypeError, AttributeError) as e:
                                # 读取错误，可能缓冲区问题或 stdout 已关闭
                                print(f"  [DEBUG] Read error: {e}")
                                line = None
                        elif self.vm_process.stderr in ready:
                            try:
                                if self.vm_process.stderr is not None:
                                    stderr_line = self.vm_process.stderr.readline()
                                    if stderr_line:
                                        # 确保是字符串类型
                                        if isinstance(stderr_line, bytes):
                                            stderr_line = stderr_line.decode('utf-8', errors='ignore')
                                        stderr_lines.append(stderr_line)
                                        # 如果是错误信息，打印出来（但不中断读取）
                                        if len(stderr_lines) <= 10:  # 只打印前10行，避免刷屏
                                            print(f"  [VM stderr] {stderr_line.strip()}")
                            except (OSError, IOError, TypeError, AttributeError) as e:
                                print(f"  [DEBUG] Stderr read error: {e}")
                            line = None
                        else:
                            line = None
                    except (OSError, ValueError) as e:
                        # 回退到阻塞读取
                        line = None
                        if self.vm_process.poll() is None and self.vm_process.stdout is not None:
                            try:
                                line = self.vm_process.stdout.readline()
                                if line:
                                    # 确保是字符串类型
                                    if isinstance(line, bytes):
                                        line = line.decode('utf-8', errors='ignore')
                                    print(f"  [DEBUG] Received line (blocking mode): {line.strip()[:100]}...")
                            except (OSError, IOError, TypeError, AttributeError) as read_err:
                                print(f"  [DEBUG] Blocking read error: {read_err}")
                                line = None
                else:
                    # 使用较短的超时，但仍然是阻塞的
                    line = None
                    if self.vm_process.poll() is None and self.vm_process.stdout is not None:
                        try:
                            line = self.vm_process.stdout.readline()
                            if line:
                                # 确保是字符串类型
                                if isinstance(line, bytes):
                                    line = line.decode('utf-8', errors='ignore')
                                print(f"  [DEBUG] Received line (blocking mode): {line.strip()[:100]}...")
                        except (OSError, IOError, TypeError, AttributeError) as read_err:
                            print(f"  [DEBUG] Blocking read error: {read_err}")
                            line = None
                
                if line:
                    line = line.strip()
                    # 跳过空行
                    if not line:
                        continue
                    # 尝试解析 JSON
                    try:
                        response = json.loads(line)
                        # 如果之前有 stderr 输出，打印出来
                        if stderr_lines:
                            print(f"VM subprocess completed with stderr output (see above)")
                        return response
                    except json.JSONDecodeError:
                        # 不是 JSON，可能是调试输出，继续读取
                        # 如果是明显的错误信息，打印出来
                        if line.startswith("Error") or line.startswith("Exception") or "Traceback" in line:
                            print(f"VM stdout (non-JSON): {line}")
                        continue
                
                # 使用更短的 sleep 时间以提高响应速度
                time.sleep(0.01)  # 从 0.1 秒减少到 0.01 秒
                wait_time += 0.01
                
                # 如果等待时间超过 5 秒还没有响应，尝试使用阻塞读取（作为后备）
                if wait_time > 5.0 and use_nonblocking:
                    print(f"  [DEBUG] Trying blocking read as fallback after {wait_time:.1f}s...")
                    try:
                        # 临时切换到阻塞模式
                        flags_backup = fcntl.fcntl(self.vm_process.stdout, fcntl.F_GETFL)
                        fcntl.fcntl(self.vm_process.stdout, fcntl.F_SETFL, flags_backup & ~os.O_NONBLOCK)
                        try:
                            # 使用 select 等待数据，最多等待 1 秒
                            if select.select([self.vm_process.stdout], [], [], 1.0)[0]:
                                line = self.vm_process.stdout.readline()
                                if line and line.strip():
                                    try:
                                        response = json.loads(line.strip())
                                        print(f"  [DEBUG] Got response via blocking fallback: {line[:100]}...")
                                        fcntl.fcntl(self.vm_process.stdout, fcntl.F_SETFL, flags_backup | os.O_NONBLOCK)
                                        return response
                                    except json.JSONDecodeError:
                                        pass
                        finally:
                            fcntl.fcntl(self.vm_process.stdout, fcntl.F_SETFL, flags_backup | os.O_NONBLOCK)
                    except Exception as e:
                        print(f"  [DEBUG] Blocking fallback failed: {e}")
                        pass
        finally:
            # 恢复阻塞模式（如果之前设置了非阻塞）
            if use_nonblocking:
                try:
                    flags = fcntl.fcntl(self.vm_process.stdout, fcntl.F_GETFL)
                    fcntl.fcntl(self.vm_process.stdout, fcntl.F_SETFL, flags & ~os.O_NONBLOCK)
                except (OSError, AttributeError):
                    pass
        
        # 超时后，如果有 stderr 输出，打印出来
        if stderr_lines:
            print(f"VM subprocess timeout. Recent stderr output:")
            for line in stderr_lines[-10:]:  # 只打印最后10行
                print(f"  {line.strip()}")
        
        return None
    
    def _base64_to_video_input(self, obs_b64):
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
    
    def _extract_task_from_prompt(self, prompt):
        """从 prompt 中提取任务描述"""
        if "performing the following task: " in prompt:
            return prompt.split("performing the following task: ")[-1].strip()
        return prompt
    
    def _tensor_to_base64_vidar(self, img_tensor):
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
    
    def get_subgoals(self, first_frame_obs_b64, instruction=None):
        """
        基于第一帧观察和instruction生成间隔8帧的subgoals
        :param first_frame_obs_b64: 第一帧观察（base64 JPEG）
        :param instruction: 任务instruction（可选，如果为None则使用self.prompt）
        :return: subgoals（list of base64 JPEG），每间隔8帧的subgoals
        """
        if not self.use_libero_subgoal:
            return []
        
        # 保存第一帧观察
        self.first_frame_obs = first_frame_obs_b64
        
        # 使用提供的instruction或self.prompt
        original_prompt = self.prompt
        if instruction is not None:
            # 临时设置prompt用于生成subgoals
            temp_prompt = instruction
        else:
            temp_prompt = self.prompt
        
        if temp_prompt is None:
            print("Warning: No instruction provided for subgoal generation")
            return []
        
        # 生成间隔8帧的subgoals
        return self.get_libero_subgoals_with_interval(first_frame_obs_b64, temp_prompt)

    def get_libero_subgoals(self, current_obs_b64):
        """
        获取 LIBERO subgoals（支持直接调用 model 或 HTTP 服务器）
        
        Args:
            current_obs_b64: 当前观察图像（base64 JPEG）
        
        Returns:
            subgoals: List[str]，subgoal 图像列表（base64 JPEG）
        """
        if not self.use_libero_subgoal:
            return []
        
        # 优先使用直接调用模式（通过 subprocess）
        if self.libero_use_direct_model and self.video_model is not None and self.vm_process is not None:
            return self._get_subgoals_direct(current_obs_b64)
        else:
            # 回退到 HTTP 服务器模式
            return self._get_subgoals_via_http(current_obs_b64)
    
    def get_libero_subgoals_with_interval(self, first_frame_obs_b64, instruction):
        """
        基于第一帧观察和instruction生成间隔8帧的subgoals
        
        Args:
            first_frame_obs_b64: 第一帧观察图像（base64 JPEG）
            instruction: 任务instruction
        
        Returns:
            subgoals: List[str]，间隔8帧的subgoal图像列表（base64 JPEG）
        """
        if not self.use_libero_subgoal:
            return []
        
        # 计算需要生成的subgoals数量（基于max_steps和间隔）
        num_subgoals = (self.max_steps // self.subgoal_interval) + 1
        # 限制最大数量，避免生成过多
        num_subgoals = min(num_subgoals, 20)
        
        # 优先使用直接调用模式（通过 subprocess）
        if self.libero_use_direct_model and self.video_model is not None and self.vm_process is not None:
            return self._get_subgoals_direct_with_interval(first_frame_obs_b64, instruction, num_subgoals)
        else:
            # 回退到 HTTP 服务器模式
            return self._get_subgoals_via_http_with_interval(first_frame_obs_b64, instruction, num_subgoals)
    
    def _get_subgoals_direct(self, current_obs_b64):
        """通过 subprocess 使用 video model 生成 subgoals"""
        try:
            if self.vm_process is None:
                raise RuntimeError("VM process not initialized")
            
            # 发送生成 subgoals 的请求（使用当前观察作为第一帧）
            generate_request = {
                'command': 'generate_subgoals',
                'first_frame_obs_b64': current_obs_b64,
                'instruction': self.prompt,
                'num_subgoals': 7,  # 默认生成7个subgoals
                'subgoal_interval': 1,  # 连续生成
                'model_path': self.libero_model_path,
                'milestone': self.libero_milestone,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            
            request_json = json.dumps(generate_request) + '\n'
            self.vm_process.stdin.write(request_json)
            self.vm_process.stdin.flush()
            
            # 读取响应（使用辅助函数，自动跳过非 JSON 行）
            response = self._read_json_response_from_vm(timeout=60)
            
            if response is not None:
                if response.get('success'):
                    subgoals = response.get('subgoals', [])
                    print(f"Generated {len(subgoals)} subgoals using LIBERO video model (subprocess mode)")
                    return subgoals
                else:
                    error_msg = response.get('error', 'Unknown error')
                    print(f"Error generating subgoals: {error_msg}")
                    # 如果直接调用失败，尝试回退到 HTTP 模式
                    if self.libero_use_direct_model:
                        print("Falling back to HTTP server mode...")
                        return self._get_subgoals_via_http(current_obs_b64)
                    return []
            else:
                print("Failed to get response from VM subprocess")
                # 如果直接调用失败，尝试回退到 HTTP 模式
                if self.libero_use_direct_model:
                    print("Falling back to HTTP server mode...")
                    return self._get_subgoals_via_http(current_obs_b64)
                return []
            
        except Exception as e:
            print(f"Error generating subgoals with video model (subprocess mode): {e}")
            import traceback
            traceback.print_exc()
            # 如果直接调用失败，尝试回退到 HTTP 模式
            if self.libero_use_direct_model:
                print("Falling back to HTTP server mode...")
                return self._get_subgoals_via_http(current_obs_b64)
            return []
    
    def _get_subgoals_direct_with_interval(self, first_frame_obs_b64, instruction, num_subgoals):
        """通过 subprocess 使用 video model 生成间隔8帧的subgoals"""
        try:
            if self.vm_process is None:
                raise RuntimeError("VM process not initialized")
            
            # 检查 subprocess 是否还在运行
            if self.vm_process.poll() is not None:
                print(f"VM subprocess has exited with code {self.vm_process.returncode}")
                # 尝试读取 stderr 获取错误信息
                try:
                    stderr_output = self.vm_process.stderr.read()
                    if stderr_output:
                        print(f"VM subprocess stderr: {stderr_output[:500]}")
                except Exception:
                    pass
                # 回退到 HTTP 模式
                if self.libero_use_direct_model:
                    print("Falling back to HTTP server mode...")
                    return self._get_subgoals_via_http_with_interval(first_frame_obs_b64, instruction, num_subgoals)
                return []
            
            # 发送生成 subgoals 的请求
            generate_request = {
                'command': 'generate_subgoals',
                'first_frame_obs_b64': first_frame_obs_b64,
                'instruction': instruction,
                'num_subgoals': num_subgoals,
                'subgoal_interval': self.subgoal_interval,
                'model_path': self.libero_model_path,  # 如果模型未加载，需要提供路径
                'milestone': self.libero_milestone,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            }
            
            request_json = json.dumps(generate_request) + '\n'
            print(f"Sending subgoal generation request: num_subgoals={num_subgoals}, interval={self.subgoal_interval}")
            self.vm_process.stdin.write(request_json)
            self.vm_process.stdin.flush()
            
            # 读取响应（增加超时时间，因为生成多个 subgoals 可能需要更长时间）
            # 每个 subgoal 可能需要几秒，所以总超时时间设置为 num_subgoals * 10 秒，最少 60 秒
            timeout = max(60, num_subgoals * 10)
            print(f"Waiting for response (timeout={timeout}s)...")
            response = self._read_json_response_from_vm(timeout=timeout)
            
            # 检查 subprocess 是否在处理过程中退出
            if self.vm_process.poll() is not None:
                print(f"VM subprocess exited during processing with code {self.vm_process.returncode}")
                # 尝试读取 stderr 获取错误信息
                try:
                    stderr_output = self.vm_process.stderr.read()
                    if stderr_output:
                        print(f"VM subprocess stderr: {stderr_output[:1000]}")
                except Exception:
                    pass
                # 回退到 HTTP 模式
                if self.libero_use_direct_model:
                    print("Falling back to HTTP server mode...")
                    return self._get_subgoals_via_http_with_interval(first_frame_obs_b64, instruction, num_subgoals)
                return []
            
            if response is not None:
                if response.get('success'):
                    subgoals = response.get('subgoals', [])
                    print(f"Generated {len(subgoals)} subgoals (interval={self.subgoal_interval} frames) using LIBERO video model (subprocess mode)")
                    return subgoals
                else:
                    error_msg = response.get('error', 'Unknown error')
                    error_type = response.get('error_type', 'Unknown')
                    print(f"Error generating subgoals: {error_type}: {error_msg}")
                    # 如果直接调用失败，尝试回退到 HTTP 模式
                    if self.libero_use_direct_model:
                        print("Falling back to HTTP server mode...")
                        return self._get_subgoals_via_http_with_interval(first_frame_obs_b64, instruction, num_subgoals)
                    return []
            else:
                print("Failed to get response from VM subprocess (timeout or no response)")
                # 尝试读取 stderr 获取可能的错误信息
                try:
                    import select
                    if select.select([self.vm_process.stderr], [], [], 0.1)[0]:
                        stderr_line = self.vm_process.stderr.readline()
                        if stderr_line:
                            print(f"VM subprocess stderr (latest): {stderr_line.strip()}")
                except Exception:
                    pass
                # 如果直接调用失败，尝试回退到 HTTP 模式
                if self.libero_use_direct_model:
                    print("Falling back to HTTP server mode...")
                    return self._get_subgoals_via_http_with_interval(first_frame_obs_b64, instruction, num_subgoals)
                return []
            
        except Exception as e:
            print(f"Error generating subgoals with interval using video model (subprocess mode): {e}")
            import traceback
            traceback.print_exc()
            # 如果直接调用失败，尝试回退到 HTTP 模式
            if self.libero_use_direct_model:
                print("Falling back to HTTP server mode...")
                return self._get_subgoals_via_http_with_interval(first_frame_obs_b64, instruction, num_subgoals)
            return []
    
    def _get_subgoals_via_http(self, current_obs_b64):
        """通过 HTTP 服务器获取 subgoals（原有方法）"""
        try:
            headers = {"Content-Type": "application/json"}
            num_subgoals = 7  # 默认生成7个subgoals
            data = {
                "prompt": self.prompt,
                "img": current_obs_b64,
                "num_subgoals": num_subgoals,
                "seed": self.seed,
                "password": "r49h8fieuwK",
            }
            
            # 根据subgoal数量动态调整超时时间：每个subgoal至少20秒，最少120秒，最多600秒（10分钟）
            timeout = min(600, max(120, num_subgoals * 20))
            print(f"Calling LIBERO subgoal server (timeout={timeout}s, num_subgoals={num_subgoals})...")
            
            response = requests.post(
                f"{self.libero_subgoal_url}/generate_subgoals",
                headers=headers,
                data=json.dumps(data),
                verify=False,
                timeout=timeout  # 动态超时时间
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    subgoals = result.get("subgoals", [])
                    print(f"Generated {len(subgoals)} subgoals from LIBERO server")
                    return subgoals
                else:
                    print(f"LIBERO subgoal generation failed: {result.get('error', 'Unknown error')}")
                    return []
            else:
                print(f"LIBERO subgoal server returned status {response.status_code}")
                return []
                
        except requests.exceptions.Timeout as e:
            print(f"Error calling LIBERO subgoal server: Read timed out after {timeout}s")
            print(f"  → The server may be processing (model inference can take 1-3 minutes)")
            print(f"  → Check server logs to see if it's still running")
            print(f"  → Consider increasing timeout or optimizing model inference")
            return []
        except requests.exceptions.ConnectionError as e:
            print(f"Error calling LIBERO subgoal server: {e}")
            print(f"  → HTTP server is not running on {self.libero_subgoal_url}")
            print(f"  → Please start the server using:")
            print(f"     cd /mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm/diffuser/libero")
            print(f"     bash start_libero_subgoal_server.sh")
            return []
        except Exception as e:
            print(f"Error calling LIBERO subgoal server: {e}")
            return []
    
    def _get_subgoals_via_http_with_interval(self, first_frame_obs_b64, instruction, num_subgoals):
        """通过 HTTP 服务器获取间隔8帧的subgoals"""
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "prompt": instruction,
                "img": first_frame_obs_b64,
                "num_subgoals": num_subgoals,
                "subgoal_interval": self.subgoal_interval,  # 传递间隔参数
                "seed": self.seed,
                "password": "r49h8fieuwK",
            }
            
            # 根据subgoal数量动态调整超时时间：每个subgoal至少20秒，最少120秒，最多600秒（10分钟）
            # 生成多个subgoals需要更长时间，特别是间隔生成模式
            timeout = min(600, max(120, num_subgoals * 20))
            print(f"Calling LIBERO subgoal server (timeout={timeout}s, num_subgoals={num_subgoals}, interval={self.subgoal_interval})...")
            
            response = requests.post(
                f"{self.libero_subgoal_url}/generate_subgoals",
                headers=headers,
                data=json.dumps(data),
                verify=False,
                timeout=timeout  # 动态超时时间，根据subgoal数量调整
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success", False):
                    subgoals = result.get("subgoals", [])
                    print(f"Generated {len(subgoals)} subgoals (interval={self.subgoal_interval} frames) from LIBERO server")
                    return subgoals
                else:
                    print(f"LIBERO subgoal generation failed: {result.get('error', 'Unknown error')}")
                    return []
            else:
                print(f"LIBERO subgoal server returned status {response.status_code}")
                return []
                
        except requests.exceptions.Timeout as e:
            print(f"Error calling LIBERO subgoal server: Read timed out after {timeout}s")
            print(f"  → The server may be processing (model inference can take 1-3 minutes)")
            print(f"  → Check server logs to see if it's still running")
            print(f"  → Consider increasing timeout or optimizing model inference")
            return []
        except requests.exceptions.ConnectionError as e:
            print(f"Error calling LIBERO subgoal server: {e}")
            print(f"  → HTTP server is not running on {self.libero_subgoal_url}")
            print(f"  → Please start the server using:")
            print(f"     cd /mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm/diffuser/libero")
            print(f"     bash start_libero_subgoal_server.sh")
            return []
        except Exception as e:
            print(f"Error calling LIBERO subgoal server: {e}")
            return []

    def get_actions(self):
        if len(self.obs_cache) >= self.max_steps:
            return []
        headers = {
            "Content-Type": "application/json",
        }
        port, seed = self.port, self.seed
        t = time.time()
        
        # 准备观察缓存
        if self.num_conditional_frames + self.num_new_frames > self.rollout_bound:
            self.num_conditional_frames = self.rollout_prefill_num
            obs_cache = self.obs_cache[-self.num_conditional_frames:]
            clean_cache = True
        else:
            obs_cache = self.obs_cache[-self.num_new_frames:]
            clean_cache = False
        
        # 如果使用 LIBERO subgoal，获取当前 subgoals
        subgoal_frames = []
        if self.use_libero_subgoal:
            # 如果还没有生成subgoals，需要先基于第一帧观察生成
            # 这应该在eval函数中完成，但这里作为fallback
            if not self.current_subgoals and self.first_frame_obs:
                # 使用第一帧观察和instruction生成subgoals
                self.current_subgoals = self.get_libero_subgoals_with_interval(
                    self.first_frame_obs, 
                    self.prompt
                )
                self.subgoal_idx = 0
            
            # 根据当前帧数选择对应的subgoal（基于间隔8帧）
            # 当前帧数 = len(self.obs_cache) - 1（因为obs_cache包含当前帧）
            current_frame = len(self.obs_cache) - 1
            subgoal_index = current_frame // self.subgoal_interval
            
            # 使用对应的subgoal
            if self.current_subgoals and subgoal_index < len(self.current_subgoals):
                subgoal_frames = [self.current_subgoals[subgoal_index]]
                print(f"Using subgoal {subgoal_index + 1}/{len(self.current_subgoals)} (frame {current_frame}, interval={self.subgoal_interval})")
            elif self.current_subgoals:
                # 如果超出范围，使用最后一个subgoal
                subgoal_frames = [self.current_subgoals[-1]]
                print(f"Using last subgoal {len(self.current_subgoals)}/{len(self.current_subgoals)} (frame {current_frame} exceeds range)")
        
        data = {
            "prompt": self.prompt, 
            "imgs": obs_cache, 
            "num_conditional_frames": self.num_conditional_frames, 
            "num_new_frames": self.num_new_frames, 
            "seed": seed, 
            "num_sampling_step": self.num_sampling_step, 
            "guide_scale": self.guide_scale, 
            "password": "r49h8fieuwK", 
            "return_imgs": True, 
            "clean_cache": clean_cache,
            "subgoal_frames": subgoal_frames,  # 传递 subgoals 给 vidar
            # MPC 参数（v2 版本）
            "use_mpc": self.use_mpc,
            "mpc_num_candidates": self.mpc_num_candidates,
            "mpc_cost_weights": self.mpc_cost_weights,
            "use_diffusion_forcing": self.use_diffusion_forcing,  # 传递 diffusion forcing 参数
        }
        
        print(f"DEBUG [AR.get_actions]: Sending request to server - use_mpc={self.use_mpc}, "
              f"mpc_num_candidates={self.mpc_num_candidates}, "
              f"num_conditional_frames={self.num_conditional_frames}, "
              f"num_new_frames={self.num_new_frames}, "
              f"obs_cache_len={len(obs_cache)}")
        
        response = worker(port, headers, data, False)
        print(f"Inference done with time usage {time.time() - t}")
        actions = json.loads(response["actions"])
        # actions = self.modify_actions(actions)
        if "imgs" in response:
            self.out_imgs += response["imgs"]
        if "masks" in response:
            self.out_masks += response["masks"]
        self.num_conditional_frames += self.num_new_frames
        return actions

    def save_videos(self):
        """保存视频，包含完善的错误检查和诊断"""
        if self.out_imgs:
            try:
                save_path = os.path.join(self.save_dir, f"episode{self.episode_id}_pred_{len(self.out_imgs)}.mp4")
                print(f"Saving prediction video to: {save_path}")
                self.set_ffmpeg(save_path)
                
                # 验证图像数据
                if not self.out_imgs:
                    print("Warning: No images to save")
                    self.close_ffmpeg()
                    return
                
                for i, v in enumerate(self.out_imgs):
                    try:
                        # 解码图像
                        img = cv2.imdecode(np.frombuffer(b64decode(v), np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            print(f"Warning: Failed to decode image {i}, skipping")
                            continue
                        
                        # 验证图像尺寸
                        h, w = img.shape[:2]
                        if h != 736 or w != 640:
                            print(f"Warning: Image {i} has size {w}x{h}, expected 640x736. Resizing...")
                            img = cv2.resize(img, (640, 736))
                        
                        # 写入数据
                        img_rgb = img[:, :, ::-1]  # BGR to RGB
                        self.video_ffmpeg.stdin.write(img_rgb.tobytes())
                    except Exception as e:
                        print(f"Error processing image {i}: {e}")
                        # 继续处理其他图像
                        continue
                
                self.close_ffmpeg()
                print(f"Successfully saved prediction video: {save_path}")
            except Exception as e:
                print(f"Error saving prediction video: {e}")
                import traceback
                traceback.print_exc()
                # 确保清理
                if self.video_ffmpeg:
                    self.close_ffmpeg()
        
        if self.out_masks:
            try:
                save_path = os.path.join(self.save_dir, f"episode{self.episode_id}_mask_{len(self.out_masks)}.mp4")
                print(f"Saving mask video to: {save_path}")
                self.set_ffmpeg(save_path)
                
                # 验证图像数据
                if not self.out_masks:
                    print("Warning: No masks to save")
                    self.close_ffmpeg()
                    return
                
                for i, v in enumerate(self.out_masks):
                    try:
                        # 解码图像
                        img = cv2.imdecode(np.frombuffer(b64decode(v), np.uint8), cv2.IMREAD_COLOR)
                        if img is None:
                            print(f"Warning: Failed to decode mask {i}, skipping")
                            continue
                        
                        # 验证图像尺寸
                        h, w = img.shape[:2]
                        if h != 736 or w != 640:
                            print(f"Warning: Mask {i} has size {w}x{h}, expected 640x736. Resizing...")
                            img = cv2.resize(img, (640, 736))
                        
                        # 写入数据
                        img_rgb = img[:, :, ::-1]  # BGR to RGB
                        self.video_ffmpeg.stdin.write(img_rgb.tobytes())
                    except Exception as e:
                        print(f"Error processing mask {i}: {e}")
                        # 继续处理其他图像
                        continue
                
                self.close_ffmpeg()
                print(f"Successfully saved mask video: {save_path}")
            except Exception as e:
                print(f"Error saving mask video: {e}")
                import traceback
                traceback.print_exc()
                # 确保清理
                if self.video_ffmpeg:
                    self.close_ffmpeg()
