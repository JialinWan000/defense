import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

def parse_r_values(value):
    return [float(item.strip()) for item in value.split(',') if item.strip()]


def plot_from_json(json_file, save_name='test.pdf', selected_r=None):
    # 1. 加载 JSON 数据
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {json_file}")
        return

    # 2. 稳健地提取 scan 数据列表
    def extract_scan_list(category_dict):
        for exp_id, content in category_dict.items():
            if isinstance(content, dict) and 'scan' in content:
                print(f"--- 找到实验数据: {exp_id} (数据量: {len(content['scan'])}) ---")
                return content['scan']
        return []

    clean_true_scan = extract_scan_list(data.get('cleanTrue', {}))
    clean_false_scan = extract_scan_list(data.get('cleanFalse', {}))

    if not clean_true_scan or not clean_false_scan:
        print("错误：无法在 JSON 中定位到有效的 'scan' 列表。")
        return

    # 3. 增强的数据提取函数 (改用列表推导式，避免 zip 迭代器问题)
    def get_plot_points(scan_data, target_r):
        # 使用 0.01 的容差匹配 r (解决 0.30000000000000004 这种精度问题)
        filtered = [item for item in scan_data if 'r' in item and abs(item['r'] - target_r) < 0.01]
        
        # 按阈值 chi 排序，防止连线混乱
        filtered.sort(key=lambda x: x.get('chi', 0))
        
        chis = [item['chi'] for item in filtered]
        pis = [item['pi_val'] for item in filtered]
        
        return chis, pis

    # 4. 创建子图
    if selected_r is None:
        selected_r = [0.0, 0.5, 1.0, 1.5, 2.0]
    n_plots = len(selected_r)
    fig, axes = plt.subplots(1, n_plots, figsize=(22, 5), sharey=True)

    color_benign = '#82AFD2' # 橙色
    color_backdoor = '#F97E6E' # 蓝色

    for i, r_val in enumerate(selected_r):
        ax = axes[i]
        
        # 提取并绘制干净模型数据
        chi_t, pi_t = get_plot_points(clean_true_scan, r_val)
        if chi_t:
            print(f"绘制 r={r_val:.1f} 的干净模型数据: {len(chi_t)} 个点")
            ax.plot(chi_t, pi_t, label='Benign model', 
                    color=color_benign, marker='o', markersize=6, linewidth=2.5)
        
        # 提取并绘制后门模型数据
        chi_f, pi_f = get_plot_points(clean_false_scan, r_val)
        if chi_f:
            print(f"绘制 r={r_val:.1f} 的后门模型数据: {len(chi_f)} 个点")
            ax.plot(chi_f, pi_f, label='Backdoor model', 
                    color=color_backdoor, marker='s', markersize=6, linewidth=2.5)
        
        # 只有存在数据时才显示图例
        if chi_t or chi_f:
            ax.legend(fontsize=10, loc='lower left')
        
        # 坐标轴格式化
        ax.set_title(f'$r = {r_val:.1f}$', fontsize=18, fontweight='bold', pad=15)
        ax.set_xlabel(r'Threshold $\chi$', fontsize=14)
        if i == 0:
            ax.set_ylabel(r'Confidence $\pi_r^{\chi}$', fontsize=16)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, linestyle='--', alpha=0.5)

    # 整体标题
    # plt.suptitle(' Noise Titration Analysis: Benign vs. Backdoor Models', 
    #              fontsize=22, y=1.05, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"\n图像处理完成。请检查文件: {save_name}")

if __name__ == '__main__':
    default_json = Path('./result/Detection/bdjscc_noise_titration_scores.json')
    fallback_json = Path('./result/Detection/noise_titration_scores.json')

    if len(sys.argv) > 1:
        input_json = Path(sys.argv[1])
    elif default_json.exists():
        input_json = default_json
    else:
        input_json = fallback_json

    if len(sys.argv) > 2:
        output_pdf = sys.argv[2]
    else:
        output_pdf = str(input_json.with_suffix('.pdf'))

    selected_r = parse_r_values(sys.argv[3]) if len(sys.argv) > 3 else None

    plot_from_json(str(input_json), save_name=output_pdf, selected_r=selected_r)
