import argparse
import logging
import os
import pty
import signal
import socket
import subprocess
import sys
import termios
import time
from typing import Optional
import torch.distributed as dist

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | Rank %(process)d | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


class ServerManager:
    """Server 进程管理器 (Context Manager)"""
    def __init__(self, script_path: str, model: str, idm: str, port: int, device_id: int, cwd: str = ".", 
                 interactive_debug: bool = True):
        """
        Args:
            script_path: 服务器脚本路径
            model: 模型路径
            idm: IDM路径
            port: 端口号
            device_id: 设备ID
            cwd: 工作目录
            interactive_debug: 是否启用交互式调试模式（pdb输出到终端）
        """
        self.script_path = os.path.abspath(script_path)
        # 显式使用 bash 调用脚本，确保能正确执行
        self.cmd = [
            "bash", self.script_path, model, idm, str(port), str(device_id), "localhost"
        ]
        self.port = port
        self.device_id = device_id
        self.cwd = cwd
        self.interactive_debug = interactive_debug
        self.process: Optional[subprocess.Popen] = None

    def _wait_for_port(self, timeout: int = 600) -> bool:  # 增加到10分钟，大模型加载需要时间
        """轮询检测端口是否开启"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            # check process is running, if faild, print std out and std err
            if self.process.poll() is not None:
                logger.error(f"Server process failed to start: {self.process.poll()=}")
                return False
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(1)
                if sock.connect_ex(('localhost', self.port)) == 0:
                    return True
            time.sleep(2)
        return False

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, stopping server...")
        raise SystemExit(f"Received signal {signum}")

    def __enter__(self):
        assert os.path.exists(self.script_path), f"Server working directory not found: {self.script_path}"

        # Register signal handlers
        self.old_sigterm = signal.signal(signal.SIGTERM, self._signal_handler)
        self.old_sigint = signal.signal(signal.SIGINT, self._signal_handler)

        try:
            logger.info(f"Starting Server on Port {self.port}...")
            # 使用 preexec_fn=os.setsid 创建新的进程组，方便后续杀掉整个进程树
            logger.info(' '.join(self.cmd))
            logger.info(f"Working directory: {os.path.abspath(self.cwd)}")
            
            # 确保 CUDA_VISIBLE_DEVICES 环境变量被传递到子进程
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = str(self.device_id)
            
            if self.interactive_debug:
                # 交互式调试模式：使用伪终端（pty）确保输出到终端，支持 pdb 交互
                logger.info("Starting server in INTERACTIVE DEBUG MODE (pdb output to terminal)")
                logger.info("WARNING: Server output will be displayed in terminal, not saved to log file")
                # 设置环境变量，告诉脚本启用交互式模式
                env['INTERACTIVE_DEBUG'] = '1'
                env['PYTHONUNBUFFERED'] = '1'
                # 使用 pty 创建伪终端，确保输出直接到终端
                master_fd, slave_fd = pty.openpty()
                # 设置终端模式
                old_settings = termios.tcgetattr(slave_fd)
                new_settings = termios.tcgetattr(slave_fd)
                new_settings[3] = new_settings[3] & ~termios.ECHO  # 禁用回显（pdb需要）
                termios.tcsetattr(slave_fd, termios.TCSANOW, new_settings)
                
                self.process = subprocess.Popen(
                    self.cmd, cwd=self.cwd,
                    stdout=slave_fd,  # 使用伪终端的从端
                    stderr=slave_fd,  # 错误也输出到伪终端
                    stdin=slave_fd,  # 输入也从伪终端读取
                    env=env,
                    preexec_fn=os.setsid  # 使用 setsid 创建新进程组
                )
                # 关闭从端（子进程已经继承）
                os.close(slave_fd)
                
                # 启动一个线程来转发伪终端的输出到实际终端
                import threading
                def forward_output():
                    try:
                        while True:
                            data = os.read(master_fd, 1024)
                            if not data:
                                break
                            sys.stdout.buffer.write(data)
                            sys.stdout.buffer.flush()
                    except (OSError, ValueError):
                        pass
                    finally:
                        os.close(master_fd)
                
                forward_thread = threading.Thread(target=forward_output, daemon=True)
                forward_thread.start()
                
                # 处理输入（从实际终端到伪终端）
                def forward_input():
                    try:
                        while True:
                            data = sys.stdin.buffer.read(1)
                            if not data:
                                break
                            os.write(master_fd, data)
                    except (OSError, ValueError, KeyboardInterrupt):
                        pass
                
                input_thread = threading.Thread(target=forward_input, daemon=True)
                input_thread.start()
                
                logger.info("Server started in interactive mode - pdb will work interactively")
                logger.info("NOTE: Server output and pdb will appear in this terminal")
            else:
                # 正常模式：输出到日志文件
                log_file = os.path.join(self.cwd, f"server_rank{self.device_id}_port{self.port}.log")
                with open(log_file, "w") as f:
                    self.process = subprocess.Popen(
                        self.cmd, cwd=self.cwd,
                        stdout=f, stderr=subprocess.STDOUT,
                        preexec_fn=os.setsid,
                        env=env
                    )
                logger.info(f"Server log file: {log_file}")

            if self._wait_for_port():
                logger.info("Server is READY.")
                return self
            
            raise RuntimeError("Server failed to start within timeout.")

        except (Exception, SystemExit):
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
            except Exception as e:
                logger.error(f"Failed to kill server process {self.process.pid}: {e}")
            

def run_client_task(task_name: str, args, port: int, output_dir_base: str, local_rank: int):
    """运行单个 Client 任务"""
    task_out_dir = os.path.join(output_dir_base, args.prefix, task_name)
    log_file = os.path.join(task_out_dir, "log.txt")
    result_file = os.path.join(task_out_dir, "_result_test.txt")

    # 自动跳过：只有当结果文件存在时才跳过（说明任务成功完成）
    if os.path.exists(result_file):
        logger.info(f"Task {task_name} already completed. Skipping.")
        return
    
    # 如果只有日志文件但没有结果文件，说明任务失败了，可以重新运行
    if os.path.exists(log_file):
        logger.info(f"Task {task_name} has log but no result file. Will retry.")

    os.makedirs(task_out_dir, exist_ok=True)
    
    # 使用 conda 环境中的 Python 解释器
    python_bin = "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/env/RoboTwin-hb/bin/python"
    cmd = [
        python_bin, "script/eval_policy.py",
        "--config", "policy/AR/deploy_policy.yml",
        "--overrides",
        "--task_name", task_name,
        "--task_config", args.task_config,
        "--port", str(port),
        "--seed", str(args.seed),
        "--policy_name", "AR",
        "--num_new_frames", str(args.num_new_frames),
        "--num_sampling_step", str(args.num_sampling_step),
        "--guide_scale", str(args.cfg),
        "--rollout_bound", str(args.rollout_bound),
        "--rollout_prefill_num", str(args.rollout_prefill_num),
        "--save_dir", task_out_dir,
        "--version", args.version if args.version else "",
        "--use_libero_subgoal", args.use_libero_subgoal,
        # "--libero_use_direct_model", args.libero_use_direct_model,
        # "--libero_model_path", args.libero_model_path,
        # "--libero_logbase", args.libero_logbase,
        # "--libero_dataset", args.libero_dataset,
        # "--libero_exp_name", args.libero_exp_name,
        # "--libero_epoch", args.libero_epoch,
        # "--libero_milestone", str(args.libero_milestone),
        "--use_vid_first_n_frames", str(args.use_vid_first_n_frames),
        "--num_vid_pred_per_ep", str(args.num_vid_pred_per_ep)
    ]
    
    # Add MPC parameters if provided (for v2_mpc version)
    if args.mpc_num_candidates is not None:
        cmd.extend(["--mpc_num_candidates", str(args.mpc_num_candidates)])
    if args.mpc_cost_weights is not None:
        cmd.extend(["--mpc_cost_weights", args.mpc_cost_weights])
    logger.info(" ".join(cmd))
    logger.info(f"Running Task: {task_name}")
    env = os.environ.copy()
    env["PYTHONWARNINGS"] = "ignore::UserWarning"
    env['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    env['PYTHONUNBUFFERED'] = '1'
    
    # 确保 PATH 包含常用路径（用于查找 ffmpeg 等工具）
    conda_env_bin = "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/env/RoboTwin-hb/bin"
    if conda_env_bin not in env.get('PATH', ''):
        env['PATH'] = conda_env_bin + ':' + env.get('PATH', '')
    
    # 调试模式：如果设置了 DEBUG_MODE 环境变量，允许交互式调试
    debug_mode = os.environ.get("DEBUG_MODE", "0") == "1"
    
    if debug_mode:
        # 调试模式：输出到终端，允许交互（确保 stdin 连接到终端）
        logger.info("DEBUG MODE: Running with interactive output (pdb will be interactive)")
        import sys
        subprocess.run(
            cmd, 
            check=False, 
            env=env,
            stdin=sys.stdin,  # 使用原始的 stdin，允许 pdb 交互
            stdout=sys.stdout,  # 输出到终端
            stderr=sys.stderr
        )
    else:
        # 正常模式：同时输出到终端和日志文件
        logger.info(f"Running task with output to terminal and log file: {log_file}")
        import threading
        
        # 使用 Popen 手动处理输出，同时写入终端和文件
        with open(log_file, 'w') as log_f:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                env=env,
                bufsize=1,  # 行缓冲
                universal_newlines=True,
                text=True
            )
            
            def tee_output():
                """在后台线程中同时写入终端和文件"""
                try:
                    for line in iter(process.stdout.readline, ''):
                        if line:
                            # 同时写入终端和文件
                            sys.stdout.write(line)
                            sys.stdout.flush()
                            log_f.write(line)
                            log_f.flush()
                except Exception as e:
                    logger.error(f"Error in tee_output thread: {e}")
            
            # 启动后台线程处理输出
            output_thread = threading.Thread(target=tee_output, daemon=True)
            output_thread.start()
            
            # 等待进程完成
            return_code = process.wait()
            
            # 等待输出线程完成（最多等待1秒）
            output_thread.join(timeout=1.0)

def run_video_model(video_model_config_path: str, local_rank: int = 0):
    """启动视频模型服务（如果需要）"""
    if not video_model_config_path or not os.path.exists(video_model_config_path):
        logger.warning(f"Video model config path not provided or not found: {video_model_config_path}. Skipping video model setup.")
        return False
    
    python_vm = "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/env/v4a/bin/python"
    if not os.path.exists(python_vm):
        logger.error(f"Video model Python interpreter not found: {python_vm}")
        return False
    
    script_path = "/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm/diffuser/libero/plan_lb.py"
    if not os.path.exists(script_path):
        logger.error(f"Video model script not found: {script_path}")
        return False
    
    cmd = [
        python_vm, script_path,
        "--config", video_model_config_path,
        "--plan_n_maze", "25",
        "--diffusion_epoch", "latest",
        "--vid_var_temp", "1.0",
    ]
    
    logger.info(f"Starting video model with config: {video_model_config_path}")
    logger.info(" ".join(cmd))
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(local_rank)
    env['PYTHONUNBUFFERED'] = '1'
    
    # 是否捕获输出（用于调试）还是实时显示在终端
    # 设置环境变量 CAPTURE_VIDEO_MODEL_OUTPUT=1 来捕获输出
    capture_output = os.environ.get("CAPTURE_VIDEO_MODEL_OUTPUT", "0") == "1"
    # 添加以下环境变量来避免交互式提示
    # 1. 设置 LIBERO 数据集路径（如果知道的话）
    env['LIBERO_DATASET_DIR'] = '/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm/datasets'  # 根据实际情况设置
    try:
        if capture_output:
            # 捕获输出模式：用于调试，输出会被保存
            logger.info("Running video model (capturing output)...")
            result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
            logger.info("Video model setup completed successfully")
            if result.stdout:
                logger.info(f"Video model stdout:\n{result.stdout}")
            if result.stderr:
                logger.info(f"Video model stderr:\n{result.stderr}")
            return True
        else:
            # 实时显示模式：输出直接显示在终端
            logger.info("Running video model (output will be displayed in terminal)...")
            result = subprocess.run(cmd, env=env, check=True)
            logger.info("Video model setup completed successfully")
            return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Video model setup failed with return code {e.returncode}")
        if capture_output and hasattr(e, 'stderr') and e.stderr:
            logger.error(f"stderr: {e.stderr}")
        if capture_output and hasattr(e, 'stdout') and e.stdout:
            logger.error(f"stdout: {e.stdout}")
        return False
    except Exception as e:
        logger.error(f"Error running video model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server_script", type=str, required=True, help="Server script path")
    parser.add_argument("--server_cwd", type=str, default="../vidar", help="Server working directory")
    parser.add_argument("--task_dir", type=str, default="./description/task_instruction", help="Task instruction directory")
    parser.add_argument("--task_name", type=str, default=None, help="Run a specific task only (optional)")
    parser.add_argument("--output_dir", type=str, default="eval_result/ar")
    parser.add_argument("--model", type=str, required=True, help="Model path for server")
    parser.add_argument("--idm", type=str, default="0418_3e-3_60000.pt", help="IDM path for server")
    parser.add_argument("--task_config", type=str, default="hd_clean", help="Task configuration")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--prefix", type=str, default="debug", help="Prefix for output directory")
    parser.add_argument("--num_new_frames", type=int, default=80)
    parser.add_argument("--num_sampling_step", type=int, default=5, help="Number of sampling steps")
    parser.add_argument("--cfg", type=float, default=5.0, help="Guide scale")
    parser.add_argument("--rollout_prefill_num", type=int, default=33)
    parser.add_argument("--rollout_bound", type=int, default=60)
    parser.add_argument("--base_port", type=int, default=25400)
    # Version parameter (overrides use_libero_subgoal if specified)
    parser.add_argument("--version", type=str, default=None, 
                       choices=["v0_original", "v1_subgoal", "v2_mpc", "df"], 
                       help="Version to use: v0_original (no subgoals), v1_subgoal (with subgoals), v2_mpc (with MPC), or df (diffusion forcing)")
    # MPC parameters (for v2_mpc version)
    parser.add_argument("--mpc_num_candidates", type=int, default=None, 
                       help="Number of MPC candidates (for v2_mpc version)")
    parser.add_argument("--mpc_cost_weights", type=str, default=None,
                       help="MPC cost weights in JSON format, e.g., '{\"task\":1.0,\"ctrl\":0.1,\"reach\":0.5}' (for v2_mpc version)")
    # Video model subgoal parameters
    parser.add_argument("--use_video_subgoals", type=str, default="true", help="Enable video model subgoals (true/false)")
    parser.add_argument("--use_libero_subgoal", type=str, default="true", help="Enable libero subgoal (true/false)")
    # parser.add_argument("--libero_use_direct_model", type=str, default="true", help="Enable libero use direct model (true/false)")
    # parser.add_argument("--libero_model_path", type=str, default="/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm/ckpts/libero/libero_ep20_bs12_aug", help="Libero model path")
    # parser.add_argument("--libero_logbase", type=str, default="/mnt/shared-storage-user/qinyiran/cyujie/cyujie/code/vidar/vm/logs", help="Libero logbase")
    # parser.add_argument("--libero_dataset", type=str, default="libero-8tk-65to72-v3", help="Libero dataset")
    # parser.add_argument("--libero_exp_name", type=str, default="lb_tk8_65to72", help="Libero exp name")
    # parser.add_argument("--libero_epoch", type=str, default="latest", help="Libero epoch")
    # parser.add_argument("--libero_milestone", type=int, default=24, help="Libero milestone")
    parser.add_argument("--use_vid_first_n_frames", type=int, default=2, help="Number of first frames to use as subgoals")
    parser.add_argument("--num_vid_pred_per_ep", type=int, default=5, help="Number of video predictions per episode")
    parser.add_argument("--video_model_config_path", type=str, default="../vidar/vm/config/libero/lb_tk8_65to72.py", help="Video model config file path (e.g., ../vidar/vm/config/libero/lb_tk8_65to72.py)")
    parser.add_argument("--interactive_debug", type=bool, default=True, help="Enable interactive debug mode (pdb output to terminal, not log file)")
    args = parser.parse_args()
    
    
    args.model = os.path.abspath(args.model)
    args.idm = os.path.abspath(args.idm)
    assert os.path.exists(args.model), f"Model path not found: {args.model}"
    assert os.path.exists(args.idm), f"IDM path not found: {args.idm}"
    # DDP 初始化
    rank, world_size, local_rank = 0, 1, 0
    if 'LOCAL_RANK' in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        logger.info("Running in single process mode (Non-DDP).")

    # 获取并切分任务
    all_tasks = sorted([i.split(".")[0] for i in os.listdir(args.task_dir)])
    all_tasks = ['stack_bowls_two', 'place_cans_plasticbox', 'beat_block_hammer', 'pick_dual_bottles', 'click_alarmclock', 'click_bell', 'shake_bottle_horizontally', 'open_laptop','turn_switch', 'press_stapler', 'shake_bottle', 'place_bread_basket', 'grab_roller', 'place_burger_fries', 'place_phone_stand', 'place_object_stand', 'place_container_plate', 'place_a2b_right', 'place_empty_cup', 'adjust_bottle', 'dump_bin_bigbin']
    
    # 如果指定了单个任务，只运行该任务
    if args.task_name:
        if args.task_name not in all_tasks:
            logger.error(f"Task '{args.task_name}' not found in task list: {all_tasks}")
            return
        my_tasks = [args.task_name]
    else:
        my_tasks = all_tasks[rank::world_size]
    
    logger.info(f"Rank {rank} {local_rank=} assigned {len(my_tasks)} tasks")

    if not my_tasks:
        return

    # 启动 Server 并运行 Client
    try:
        # 如果启用了视频子目标，先启动视频模型（如果需要）
        if args.use_video_subgoals == True:
            # 如果 video_model_config_path 为空字符串，使用 argparse 的默认值
            config_path = args.video_model_config_path.strip() if args.video_model_config_path and args.video_model_config_path.strip() else "../vidar/vm/config/libero/lb_tk8_65to72.py"
            logger.info("Video subgoals enabled, setting up video model...")
            logger.info(f"Using video model config: {config_path}")
            run_video_model(config_path, local_rank)
        
        with ServerManager(
            args.server_script, args.model, args.idm, 
            args.base_port + rank, local_rank, args.server_cwd,
            interactive_debug=args.interactive_debug
        ):
            for task in my_tasks:
                run_client_task(task, args, args.base_port + rank, args.output_dir, local_rank)
    except Exception as e:
        logger.error(f"Error in Rank {rank}: {e}")
        if world_size > 1:
            raise e



if __name__ == "__main__":
    main()
