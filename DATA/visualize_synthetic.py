"""
合成 EOG 数据可视化：每类展示 3 个样本的波形
使用方法：
    cd DATA
    python visualize_synthetic.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

SYNTHETIC_DIR = "synthetic_EOG"
CLASSES = ["Rest", "Up", "Down", "Left", "Right", "Blink"]
SAMPLES_TO_SHOW = 3  # 每类展示几个样本

fig, axes = plt.subplots(len(CLASSES), SAMPLES_TO_SHOW, figsize=(15, 14))
fig.suptitle("合成 EOG 数据波形预览（蓝=H通道, 红=V通道）", fontsize=14, y=0.98)

for row, cls in enumerate(CLASSES):
    cls_dir = os.path.join(SYNTHETIC_DIR, cls)
    csv_files = sorted(f for f in os.listdir(cls_dir) if f.endswith('.csv'))

    for col in range(SAMPLES_TO_SHOW):
        ax = axes[row][col]
        df = pd.read_csv(os.path.join(cls_dir, csv_files[col]))
        t = df['Program Time [s]'].values
        h = df['data 0'].values
        v = df['data 1'].values

        ax.plot(t, h, 'b-', linewidth=1.2, label='H')
        ax.plot(t, v, 'r-', linewidth=1.2, alpha=0.8, label='V')
        ax.set_ylim(200, 950)
        ax.grid(True, alpha=0.3)

        if col == 0:
            ax.set_ylabel(cls, fontsize=12, fontweight='bold')
        if row == 0:
            ax.set_title(f"样本 {col+1}", fontsize=10)
            if col == 0:
                ax.legend(loc='upper right', fontsize=8)
        if row == len(CLASSES) - 1:
            ax.set_xlabel("Time (s)")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("synthetic_preview.png", dpi=150)
plt.show()
print("已保存: synthetic_preview.png")
