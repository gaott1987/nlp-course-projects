import json
import glob
import matplotlib.pyplot as plt
import sys
import os

# 导入已有的评估函数
from evaluate import eval_single

def parse_pred_filename(filename):
    """解析预测文件名"""
    basename = os.path.basename(filename).replace('.json', '')
    parts = basename.split('_')
    
    epoch = None
    step = None
    for i, part in enumerate(parts):
        if part == 'epoch' and i + 1 < len(parts):
            epoch = int(parts[i + 1])
        if part.isdigit() and i > 0 and parts[i-1] != 'epoch':
            step = int(part)
    
    return epoch, step


def collect_results(pred_dir, gold_file, pattern='pred_seed_*_epoch_*_99999.json'):
    """只收集每个epoch结束的预测文件"""
    pred_files = glob.glob(f"{pred_dir}/{pattern}")
    results = []
    
    print(f"找到 {len(pred_files)} 个预测文件\n")
    
    for pred_file in pred_files:  # 先不排序，后面统一按epoch排序
        epoch, step = parse_pred_filename(pred_file)
        if epoch is None:
            print(f"警告: 无法解析文件名 {pred_file}，跳过")
            continue
        
        print(f"评估: Epoch {epoch}, Step {step}")
        
        # 调用eval_single函数
        metrics = eval_single(pred_file, gold_file)
        
        results.append({
            'epoch': epoch,
            'step': step,
            'file': pred_file,
            'metrics': metrics
        })
    
    # 关键修改1: 按epoch从小到大排序
    results.sort(key=lambda x: x['epoch'])
    
    return results


def plot_curves(results, save_path='metrics_curve.png'):
    """绘制评估曲线"""
    if not results:
        print("没有数据可绘图")
        return
    
    # 提取数据
    x_labels = []
    em_ans = []
    f1_ans = []
    em_sp = []
    f1_sp = []
    em_joint = []
    f1_joint = []
    
    for r in results:
        m = r['metrics']
        # 关键修改2: 简化标签，因为step都是99999，直接显示Epoch号
        label = f"E{r['epoch']}"
        x_labels.append(label)
        em_ans.append(m['em'] * 100)
        f1_ans.append(m['f1'] * 100)
        em_sp.append(m['sp_em'] * 100)
        f1_sp.append(m['sp_f1'] * 100)
        em_joint.append(m['joint_em'] * 100)
        f1_joint.append(m['joint_f1'] * 100)
    
    x = range(len(results))
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Model Performance at Epoch End', fontsize=16)
    
    axes[0,0].plot(x, em_ans, 'b-o', linewidth=2, markersize=8, label='EM')
    axes[0,0].plot(x, f1_ans, 'r-s', linewidth=2, markersize=8, label='F1')
    axes[0,0].set_title('Answer Prediction')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    axes[0,1].plot(x, em_sp, 'g-o', linewidth=2, markersize=8, label='SP EM')
    axes[0,1].plot(x, f1_sp, 'm-s', linewidth=2, markersize=8, label='SP F1')
    axes[0,1].set_title('Supporting Facts')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    axes[1,0].plot(x, em_joint, 'c-o', linewidth=2, markersize=8, label='Joint EM')
    axes[1,0].plot(x, f1_joint, 'orange', marker='s', linewidth=2, markersize=8, label='Joint F1')
    axes[1,0].set_title('Joint Metrics')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    axes[1,1].plot(x, f1_ans, 'r-o', linewidth=2, markersize=8, label='Answer F1')
    axes[1,1].plot(x, f1_sp, 'g-s', linewidth=2, markersize=8, label='SP F1')
    axes[1,1].plot(x, f1_joint, 'orange', marker='^', linewidth=2, markersize=8, label='Joint F1')
    axes[1,1].set_title('F1 Score Comparison')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0, ha='center')  # epoch数字居中显示
        ax.set_ylabel('Score (%)')
        ax.set_xlabel('Epoch')
        ax.set_ylim(0, 105)  # 设置y轴范围0-100%
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # 打印最佳结果
    print("\n" + "="*50)
    print("最佳结果汇总")
    print("="*50)
    
    best_f1 = max(results, key=lambda x: x['metrics']['f1'])
    best_em = max(results, key=lambda x: x['metrics']['em'])
    best_sp = max(results, key=lambda x: x['metrics']['sp_f1'])
    best_joint = max(results, key=lambda x: x['metrics']['joint_f1'])
    
    print(f"最佳 Answer F1:  {best_f1['metrics']['f1']*100:.2f}% (Epoch {best_f1['epoch']})")
    print(f"最佳 Answer EM:   {best_em['metrics']['em']*100:.2f}% (Epoch {best_em['epoch']})")
    print(f"最佳 SP F1:       {best_sp['metrics']['sp_f1']*100:.2f}% (Epoch {best_sp['epoch']})")
    print(f"最佳 Joint F1:    {best_joint['metrics']['joint_f1']*100:.2f}% (Epoch {best_joint['epoch']})")
    
    # 打印每个epoch的结果
    print("\n--- 各Epoch详细结果 ---")
    for r in results:
        m = r['metrics']
        print(f"Epoch {r['epoch']}: Answer F1={m['f1']*100:.2f}%, "
              f"SP F1={m['sp_f1']*100:.2f}%, "
              f"Joint F1={m['joint_f1']*100:.2f}%")


if __name__ == '__main__':
    # 使用示例:
    # python plot_curves.py
    # python plot_curves.py ./predictions ./data/dev.json
    
    PRED_DIR = sys.argv[1] if len(sys.argv) > 1 else "./predictions"
    GOLD_FILE = sys.argv[2] if len(sys.argv) > 2 else "./data/dev.json"
    
    print(f"预测文件目录: {PRED_DIR}")
    print(f"Gold文件路径: {GOLD_FILE}")
    print("-" * 50)
    
    results = collect_results(PRED_DIR, GOLD_FILE)
    
    if results:
        # 打印排序后的epoch顺序
        epochs = [r['epoch'] for r in results]
        print(f"按Epoch顺序: {epochs}")
        print("-" * 50)
        
        plot_curves(results)
    else:
        print("未找到有效的预测文件，请检查路径和文件格式")
        sys.exit(1)