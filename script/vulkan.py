#!/usr/bin/env python
"""测试 Vulkan 和 scene.update_render()"""
import os
import sys

# 设置环境变量
os.environ['DISPLAY'] = ':99'
os.environ['VK_ICD_FILENAMES'] = '/usr/share/vulkan/icd.d/lvp_icd.x86_64.json'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/nvidia/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['XDG_RUNTIME_DIR'] = '/tmp/runtime-root'

print('=' * 80)
print('测试 Vulkan 和 scene.update_render()')
print('=' * 80)

# 1. 测试 Vulkan 是否可用
print('\n[1] 检查 Vulkan 环境:')
print(f"   VK_ICD_FILENAMES: {os.environ.get('VK_ICD_FILENAMES', '未设置')}")
print(f"   DISPLAY: {os.environ.get('DISPLAY', '未设置')}")

# 检查 Vulkan ICD 文件是否存在
vk_icd = '/usr/share/vulkan/icd.d/lvp_icd.x86_64.json'
if os.path.exists(vk_icd):
    print(f"   ✓ Vulkan ICD 文件存在: {vk_icd}")
else:
    print(f"   ✗ Vulkan ICD 文件不存在: {vk_icd}")
    # 尝试查找其他 ICD 文件
    import glob
    icd_files = glob.glob('/usr/share/vulkan/icd.d/*.json')
    if icd_files:
        print(f"   找到其他 ICD 文件: {icd_files}")

# 2. 测试 SAPIEN 导入
print('\n[2] 测试 SAPIEN 导入:')
try:
    import sapien.core as sapien
    from sapien.render import set_global_config
    print('   ✓ SAPIEN 导入成功')
except Exception as e:
    print(f'   ✗ SAPIEN 导入失败: {e}')
    sys.exit(1)

# 3. 测试 Scene 创建和 update_render()
print('\n[3] 测试 Scene 创建和 update_render():')
try:
    # 设置全局配置
    set_global_config(max_num_materials=50000, max_num_textures=50000)
    print('   ✓ 全局配置设置成功')
    
    # 创建 Scene
    scene = sapien.Scene()
    print('   ✓ Scene 创建成功')
    
    # 添加一些基本内容（可选）
    scene.add_ground(0)
    scene.set_ambient_light([0.5, 0.5, 0.5])
    print('   ✓ 场景内容添加成功')
    
    # 运行物理步进
    scene.step()
    print('   ✓ Scene.step() 成功')
    
    # 测试 update_render()
    scene.update_render()
    print('   ✓ scene.update_render() 成功！')
    
    print('\n' + '=' * 80)
    print('✓✓✓ Vulkan 和 scene.update_render() 测试通过！')
    print('=' * 80)
    
except Exception as e:
    print(f'   ✗ 测试失败: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)