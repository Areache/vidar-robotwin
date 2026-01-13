# FFmpeg "Error during demuxing: Immediate exit requested" 错误诊断

## 可能的原因和解决方案

### 1. **路径问题 (Incorrect URL or Path)**

#### 问题：
- 保存路径无效、不存在或没有写权限
- 路径包含特殊字符
- 路径太长（超过系统限制）

#### 解决方案：
- ✅ 已添加路径验证：检查目录是否存在，自动创建
- ✅ 已添加写权限测试：在启动 ffmpeg 前测试文件写入
- ✅ 已添加路径长度警告

#### 检查方法：
```python
# 在 pdb 中检查
(Pdb) p model.save_dir
(Pdb) import os
(Pdb) p os.path.exists(model.save_dir)
(Pdb) p os.access(model.save_dir, os.W_OK)
```

---

### 2. **FFmpeg 可执行文件问题**

#### 问题：
- ffmpeg 不在 PATH 中
- ffmpeg 版本不兼容
- ffmpeg 权限不足

#### 解决方案：
- ✅ 已添加 ffmpeg 存在性检查
- ✅ 使用 `shutil.which()` 查找 ffmpeg
- ✅ 提供明确的错误信息

#### 检查方法：
```bash
which ffmpeg
ffmpeg -version
```

---

### 3. **图像数据问题**

#### 问题：
- 图像数据为空或格式错误
- 图像尺寸不匹配（不是 640x736）
- Base64 解码失败

#### 解决方案：
- ✅ 已添加图像解码验证
- ✅ 已添加图像尺寸检查和自动调整
- ✅ 已添加单个图像错误处理（跳过损坏的图像，继续处理其他）

#### 检查方法：
```python
# 在 pdb 中检查
(Pdb) p len(model.out_imgs)
(Pdb) import cv2, numpy as np
(Pdb) from base64 import b64decode
(Pdb) img = cv2.imdecode(np.frombuffer(b64decode(model.out_imgs[0]), np.uint8), cv2.IMREAD_COLOR)
(Pdb) p img.shape if img is not None else "Failed to decode"
```

---

### 4. **进程管理问题**

#### 问题：
- stdin 被意外关闭
- 进程被信号中断（SIGINT, SIGTERM）
- 资源不足（内存、文件描述符）

#### 解决方案：
- ✅ 已添加 stderr 捕获，可以读取详细错误信息
- ✅ 已添加进程状态检查
- ✅ 已添加优雅关闭机制（SIGTERM -> SIGKILL）

#### 检查方法：
```python
# 在 pdb 中检查
(Pdb) check_resources
(Pdb) p model.video_ffmpeg.poll() if model.video_ffmpeg else None
```

---

### 5. **FFmpeg 参数问题**

#### 问题：
- 视频尺寸参数错误
- 像素格式不匹配
- 编码器不可用

#### 解决方案：
- ✅ 已添加图像尺寸自动调整
- ✅ 使用标准参数（libx264, yuv420p）
- ✅ 捕获 stderr 以获取 ffmpeg 的详细错误信息

#### 检查方法：
```bash
# 测试 ffmpeg 命令
ffmpeg -f rawvideo -pixel_format rgb24 -video_size 640x736 -framerate 10 -i - -pix_fmt yuv420p -vcodec libx264 -crf 23 test.mp4
```

---

### 6. **并发/多进程问题**

#### 问题：
- 多个 ffmpeg 进程同时运行
- 文件被其他进程锁定
- 资源竞争

#### 解决方案：
- ✅ 已添加进程状态检查
- ✅ 在启动新进程前关闭旧进程
- ✅ 使用文件锁（通过 os.makedirs 的原子性）

---

## 诊断工具

### 在代码中添加的诊断功能：

1. **路径验证**：
   - 检查目录是否存在
   - 测试写权限
   - 验证路径长度

2. **FFmpeg 检查**：
   - 检查 ffmpeg 是否存在
   - 验证进程启动
   - 捕获 stderr 错误信息

3. **数据验证**：
   - 验证图像数据
   - 检查图像尺寸
   - 自动调整尺寸

4. **错误报告**：
   - 详细的错误信息
   - 堆栈跟踪
   - 进程状态信息

---

## 常见错误信息对照

| 错误信息 | 可能原因 | 解决方案 |
|---------|---------|---------|
| "Error during demuxing: Immediate exit requested" | stdin 被关闭、进程被中断 | 检查进程状态，确保正常关闭 |
| "No such file or directory" | 路径不存在 | 检查 save_dir，确保目录存在 |
| "Permission denied" | 没有写权限 | 检查目录权限 |
| "ffmpeg not found" | ffmpeg 不在 PATH | 安装 ffmpeg 或添加到 PATH |
| "Invalid data found when processing input" | 图像数据格式错误 | 检查图像解码和尺寸 |

---

## 调试建议

1. **启用详细日志**：
   - 代码已添加 print 语句输出关键信息
   - 检查保存路径、图像数量、进程状态

2. **使用 pdb 命令**：
   ```python
   (Pdb) check_resources  # 检查资源状态
   (Pdb) p model.save_dir  # 检查保存目录
   (Pdb) p len(model.out_imgs)  # 检查图像数量
   ```

3. **检查 ffmpeg 输出**：
   - 代码已捕获 stderr
   - 错误信息会在 close_ffmpeg 时打印

4. **测试 ffmpeg 命令**：
   - 手动运行 ffmpeg 命令测试
   - 验证参数是否正确

---

## 预防措施

1. ✅ 在启动 ffmpeg 前验证所有参数
2. ✅ 捕获并报告所有错误
3. ✅ 优雅地处理异常情况
4. ✅ 提供详细的错误信息
5. ✅ 自动调整图像尺寸
6. ✅ 验证文件系统权限

