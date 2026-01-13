"""
调试工具集合
用于在子进程和分布式环境中进行交互式调试
"""
import os
import sys
import socket
import threading
from typing import Optional

# ========== 方案1: 远程调试器 (推荐) ==========
def setup_remote_pdb(port: int = 4444, host: str = '0.0.0.0'):
    """
    设置远程调试器，可以通过 telnet 连接进行调试
    
    使用方法:
    1. 在代码中调用: setup_remote_pdb(port=4444)
    2. 在另一个终端: telnet localhost 4444
    3. 开始调试
    
    Args:
        port: 调试端口
        host: 监听地址
    """
    try:
        import rpdb
        rpdb.set_trace(port=port, host=host)
        print(f"Remote debugger started on {host}:{port}")
        print(f"Connect with: telnet {host} {port}")
    except ImportError:
        print("rpdb not installed. Install with: pip install rpdb")
        # 回退到标准 pdb
        import pdb
        pdb.set_trace()


def setup_remote_pdb2(port: int = 4444):
    """
    使用 remote-pdb (更现代的远程调试器)
    
    使用方法:
    1. 在代码中调用: setup_remote_pdb2(port=4444)
    2. 在浏览器打开: http://localhost:4444
    3. 或使用 telnet: telnet localhost 4444
    
    Args:
        port: 调试端口
    """
    try:
        from remote_pdb import RemotePdb
        RemotePdb(host='0.0.0.0', port=port).set_trace()
        print(f"Remote debugger started on port {port}")
        print(f"Connect with: telnet localhost {port} or http://localhost:{port}")
    except ImportError:
        print("remote-pdb not installed. Install with: pip install remote-pdb")
        setup_remote_pdb(port)


# ========== 方案2: 条件断点 ==========
def conditional_breakpoint(condition: bool, port: int = 4444):
    """
    条件断点：只在满足条件时触发
    
    Args:
        condition: 触发条件
        port: 远程调试端口
    """
    if condition:
        print(f"Condition met! Starting debugger on port {port}")
        setup_remote_pdb2(port)
    else:
        print(f"Condition not met, skipping breakpoint")


# ========== 方案3: 调试日志 ==========
class DebugLogger:
    """增强的调试日志记录器"""
    
    def __init__(self, log_file: Optional[str] = None, enable_interactive: bool = False):
        self.log_file = log_file
        self.enable_interactive = enable_interactive
        self.log_buffer = []
    
    def log(self, message: str, level: str = "INFO"):
        """记录日志"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] [{level}] {message}"
        
        # 输出到控制台
        print(log_msg)
        
        # 保存到缓冲区
        self.log_buffer.append(log_msg)
        
        # 写入文件
        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(log_msg + "\n")
    
    def inspect(self, obj, name: str = "object"):
        """检查对象并记录详细信息"""
        import inspect
        self.log(f"=== Inspecting {name} ===")
        self.log(f"Type: {type(obj)}")
        self.log(f"Value: {obj}")
        
        if hasattr(obj, '__dict__'):
            self.log(f"Attributes: {list(obj.__dict__.keys())}")
        
        if inspect.ismethod(obj) or inspect.isfunction(obj):
            try:
                sig = inspect.signature(obj)
                self.log(f"Signature: {sig}")
            except:
                pass
    
    def breakpoint(self, message: str = "Breakpoint reached", port: int = 4444):
        """带消息的断点"""
        self.log(f"BREAKPOINT: {message}", "DEBUG")
        if self.enable_interactive:
            setup_remote_pdb2(port)


# ========== 方案4: 环境变量控制的调试 ==========
def debug_if_enabled(port: int = 4444):
    """
    通过环境变量控制是否启用调试
    
    使用方法:
    export ENABLE_DEBUG=1
    python script.py
    """
    if os.environ.get("ENABLE_DEBUG", "0") == "1":
        print(f"Debug mode enabled. Starting debugger on port {port}")
        setup_remote_pdb2(port)
    else:
        print("Debug mode disabled. Set ENABLE_DEBUG=1 to enable.")


# ========== 方案5: 文件触发调试 ==========
def debug_if_file_exists(trigger_file: str = "/tmp/debug_trigger", port: int = 4444):
    """
    通过创建文件来触发调试
    
    使用方法:
    1. 在代码中调用: debug_if_file_exists()
    2. 需要调试时: touch /tmp/debug_trigger
    3. 调试完成后: rm /tmp/debug_trigger
    """
    if os.path.exists(trigger_file):
        print(f"Debug trigger file found: {trigger_file}")
        print(f"Starting debugger on port {port}")
        setup_remote_pdb2(port)
        # 删除触发文件，避免重复触发
        try:
            os.remove(trigger_file)
        except:
            pass


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 示例1: 远程调试
    print("Example 1: Remote debugging")
    # setup_remote_pdb2(port=4444)
    
    # 示例2: 条件断点
    print("Example 2: Conditional breakpoint")
    # conditional_breakpoint(condition=True, port=4444)
    
    # 示例3: 调试日志
    print("Example 3: Debug logger")
    logger = DebugLogger(log_file="/tmp/debug.log", enable_interactive=False)
    logger.log("Test message")
    logger.inspect([1, 2, 3], "my_list")
    
    # 示例4: 环境变量控制
    print("Example 4: Environment variable control")
    # debug_if_enabled(port=4444)
    
    # 示例5: 文件触发
    print("Example 5: File trigger")
    # debug_if_file_exists(trigger_file="/tmp/debug_trigger", port=4444)




