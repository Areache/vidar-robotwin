#!/usr/bin/env python3
"""
查看评估结果的脚本
用法: python view_results.py [结果目录]
"""

import os
import sys
from pathlib import Path
from collections import defaultdict

def parse_result_file(result_file):
    """解析结果文件"""
    if not os.path.exists(result_file):
        return None
    
    with open(result_file, 'r') as f:
        lines = f.readlines()
    
    result = {
        'timestamp': '',
        'instruction_type': '',
        'success_rates': []
    }
    
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith('Timestamp:'):
            result['timestamp'] = line.replace('Timestamp:', '').strip()
        elif line.startswith('Instruction Type:'):
            result['instruction_type'] = line.replace('Instruction Type:', '').strip()
        elif line and not line.startswith('#') and line.replace('.', '').replace('\n', '').isdigit():
            try:
                result['success_rates'].append(float(line))
            except ValueError:
                pass
    
    return result

def view_results(base_dir):
    """查看所有任务的结果"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"错误: 目录不存在: {base_dir}")
        return
    
    # 查找所有任务目录
    task_dirs = [d for d in base_path.iterdir() if d.is_dir()]
    
    if not task_dirs:
        print(f"未找到任务目录 in {base_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"评估结果汇总: {base_dir}")
    print(f"{'='*80}\n")
    
    results_summary = []
    
    for task_dir in sorted(task_dirs):
        task_name = task_dir.name
        result_file = task_dir / "_result.txt"
        log_file = task_dir / "log.txt"
        
        # 检查任务状态
        if result_file.exists():
            result = parse_result_file(result_file)
            if result and result['success_rates']:
                avg_success = sum(result['success_rates']) / len(result['success_rates'])
                max_success = max(result['success_rates'])
                min_success = min(result['success_rates'])
                
                results_summary.append({
                    'task': task_name,
                    'avg': avg_success,
                    'max': max_success,
                    'min': min_success,
                    'count': len(result['success_rates']),
                    'status': '完成'
                })
            else:
                results_summary.append({
                    'task': task_name,
                    'avg': None,
                    'max': None,
                    'min': None,
                    'count': 0,
                    'status': '结果文件为空'
                })
        elif log_file.exists():
            # 检查日志文件是否有错误
            with open(log_file, 'r') as f:
                log_content = f.read()
                if 'Traceback' in log_content or 'Error' in log_content:
                    results_summary.append({
                        'task': task_name,
                        'avg': None,
                        'max': None,
                        'min': None,
                        'count': 0,
                        'status': '运行错误'
                    })
                else:
                    results_summary.append({
                        'task': task_name,
                        'avg': None,
                        'max': None,
                        'min': None,
                        'count': 0,
                        'status': '运行中'
                    })
        else:
            results_summary.append({
                'task': task_name,
                'avg': None,
                'max': None,
                'min': None,
                'count': 0,
                'status': '未开始'
            })
    
    # 打印汇总表格
    print(f"{'任务名称':<30} {'状态':<12} {'平均成功率':<12} {'最高':<10} {'最低':<10} {'次数':<8}")
    print("-" * 80)
    
    completed_tasks = []
    for r in results_summary:
        if r['status'] == '完成':
            print(f"{r['task']:<30} {r['status']:<12} {r['avg']*100:>10.1f}% {r['max']*100:>8.1f}% {r['min']*100:>8.1f}% {r['count']:>6}")
            completed_tasks.append(r)
        else:
            print(f"{r['task']:<30} {r['status']:<12} {'-':<12} {'-':<10} {'-':<10} {'-':<8}")
    
    # 打印总体统计
    if completed_tasks:
        print("\n" + "-" * 80)
        overall_avg = sum(r['avg'] for r in completed_tasks) / len(completed_tasks)
        print(f"\n总体平均成功率: {overall_avg*100:.1f}%")
        print(f"完成任务数: {len(completed_tasks)}/{len(results_summary)}")
    
    # 打印详细结果（可选）
    print(f"\n{'='*80}")
    print("详细结果文件位置:")
    print(f"{'='*80}\n")
    for task_dir in sorted(task_dirs):
        result_file = task_dir / "_result.txt"
        if result_file.exists():
            print(f"  {task_dir.name}: {result_file}")
            result = parse_result_file(result_file)
            if result:
                print(f"    时间戳: {result['timestamp']}")
                print(f"    指令类型: {result['instruction_type']}")
                if result['success_rates']:
                    print(f"    成功率: {', '.join(f'{r*100:.1f}%' for r in result['success_rates'])}")
                print()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        # 默认结果目录
        base_dir = "eval_result/ar/ddp_causal"
    
    view_results(base_dir)

