"""Generate an improved confusion matrix figure: row-normalized + Macro-F1 + Kappa"""
import os, csv, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(BASE, "论文", "figures")  # NOTE: path kept unchanged ("论文" = "thesis") to match the actual repository directory structure
os.makedirs(OUT, exist_ok=True)

# ─── Confusion matrix for the main model (from report.md) ───
classes = ['idle', 'forehand', 'backhand', 'serve', 'move']
cm = np.array([
    [10917,    45,    78,   393,   399],   # idle
    [  127,   463,    87,     2,   106],   # forehand
    [  209,    31,   417,     1,   112],   # backhand
    [   98,     0,     0,   710,     3],   # serve
    [  651,    79,   100,    17,  2300],   # move
], dtype=float)

row_totals = cm.sum(axis=1)
col_totals = cm.sum(axis=0)
total = cm.sum()

# ─── Per-class metrics ───
per_class = {}
for i, c in enumerate(classes):
    tp = cm[i, i]
    fp = col_totals[i] - tp
    fn = row_totals[i] - tp
    p = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    r = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    per_class[c] = {'P': p, 'R': r, 'F1': f1}

macro_f1 = np.mean([v['F1'] for v in per_class.values()])
p0 = np.trace(cm) / total
pe = sum(row_totals[i] * col_totals[i] for i in range(len(classes))) / (total * total)
kappa = (p0 - pe) / (1 - pe)

# ─── Row-normalized confusion matrix ───
cm_row = cm / row_totals[:, None] * 100

# ─── Figure 1: row-normalized confusion matrix + metrics ───
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm_row, cmap='Blues', vmin=0, vmax=100)

ax.set_xticks(range(5))
ax.set_yticks(range(5))
ax.set_xticklabels(classes, fontsize=9)
ax.set_yticklabels(classes, fontsize=9)
ax.set_xlabel('Predicted', fontsize=10)
ax.set_ylabel('Ground Truth', fontsize=10)

# Numbers inside each cell
for i in range(5):
    for j in range(5):
        v = cm_row[i, j]
        if v > 80:
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        else:
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center', fontsize=8, color='black')

ax.set_title(f'Normalized Confusion Matrix — Main Model', fontsize=11, fontweight='bold')

# Additional metrics box
stats_text = (
    f'Accuracy: {p0*100:.2f}%\n'
    f'Macro-F1: {macro_f1:.2f}%\n'
    f"Cohen's Kappa: {kappa:.3f}"
)
ax.text(1.35, 0.5, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='center', horizontalalignment='left',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.8))

# Small per-class metrics table on the right
for idx, c in enumerate(classes):
    ax.text(1.35, 0.84 - idx*0.075, f'{c}:  P={per_class[c]["P"]:.1f}%  R={per_class[c]["R"]:.1f}%  F1={per_class[c]["F1"]:.1f}%',
            transform=ax.transAxes, fontsize=7, verticalalignment='center')

fig.tight_layout()
plt.savefig(os.path.join(OUT, 'fig7_confusion_matrix_main.png'), dpi=200, bbox_inches='tight')
plt.close()
print(f"fig7 saved. Macro-F1: {macro_f1:.2f}%, Kappa: {kappa:.3f}")

# ─── Figure 2: column-normalized confusion matrix ───
cm_col = cm / col_totals[None, :] * 100
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_col, cmap='Oranges', vmin=0, vmax=100)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(classes, fontsize=9); ax.set_yticklabels(classes, fontsize=9)
ax.set_xlabel('Predicted', fontsize=10); ax.set_ylabel('Ground Truth', fontsize=10)
for i in range(5):
    for j in range(5):
        v = cm_col[i, j]
        if v > 50:
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center', fontsize=8, color='white', fontweight='bold')
        else:
            ax.text(j, i, f'{v:.1f}%', ha='center', va='center', fontsize=8, color='black')
ax.set_title('Column-Normalized Confusion Matrix\n(Precision View)', fontsize=11, fontweight='bold')
fig.tight_layout()
plt.savefig(os.path.join(OUT, 'fig7_col_normalized_confusion.png'), dpi=200, bbox_inches='tight')
plt.close()
print("Column-normalized confusion matrix saved.")