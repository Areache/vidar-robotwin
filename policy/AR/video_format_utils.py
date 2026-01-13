# -- coding: UTF-8
"""
图像格式转换工具函数
用于在vidar-robotwin和video-to-action-release之间转换图像格式
"""
import cv2
import numpy as np
import torch
from base64 import b64encode, b64decode
import torchvision


def convert_obs_to_video_input(obs_b64):
    """
    将vidar的base64图像转换为video model输入格式
    
    Args:
        obs_b64: base64编码的JPEG图像字符串（vidar格式）
    
    Returns:
        img_tensor: (1, 3, 128, 128) torch tensor, RGB格式, 值域[0, 1]
    """
    # 1. 解码base64
    img_bytes = b64decode(obs_b64)
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    
    # 2. BGR -> RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 3. Resize to 128x128 (video model期望的尺寸)
    img_resized = cv2.resize(img_rgb, (128, 128))
    
    # 4. Normalize to [0, 1] and convert to tensor
    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 3, 128, 128)
    
    return img_tensor


def convert_subgoal_to_vidar_format(subgoal_tensor, target_size=(640, 736)):
    """
    将video model输出的subgoal转换为vidar可用的base64格式
    
    Args:
        subgoal_tensor: (1, 3, H, W) 或 (3, H, W) torch tensor, RGB格式, 值域[0, 1]
        target_size: 目标尺寸 (width, height)，默认(640, 736)匹配vidar输入
    
    Returns:
        img_b64: base64编码的JPEG图像字符串（vidar格式，BGR）
    """
    # 确保是4D tensor
    if subgoal_tensor.dim() == 3:
        subgoal_tensor = subgoal_tensor.unsqueeze(0)
    
    # 1. Denormalize and convert to numpy
    img = (subgoal_tensor[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    
    # 2. Resize to target size (如果需要)
    if img.shape[:2] != (target_size[1], target_size[0]):
        img_resized = cv2.resize(img, target_size)
    else:
        img_resized = img
    
    # 3. RGB -> BGR (vidar使用BGR格式)
    img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    
    # 4. Encode to base64 JPEG (匹配vidar的编码方式)
    # 使用torchvision的方式编码，与vidar保持一致
    img_tensor = torch.from_numpy(img_bgr).permute(2, 0, 1)  # (3, H, W)
    jpeg_tensor = torchvision.io.encode_jpeg(img_tensor)
    img_b64 = b64encode(jpeg_tensor.numpy().tobytes()).decode('utf-8')
    
    return img_b64


def extract_task_description(instruction):
    """
    从vidar的instruction中提取简洁的任务描述（用于video model）
    
    Args:
        instruction: vidar格式的完整指令字符串
    
    Returns:
        task_str: 简洁的任务描述字符串
    """
    if "performing the following task: " in instruction:
        task = instruction.split("performing the following task: ")[-1]
        return task.strip()
    return instruction

