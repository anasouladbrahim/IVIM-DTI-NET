import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})
 
# ── Data ───────────────────────────────────────────────────────────────────
models = ['Model 1', 'Model 2', 'Model 3', 'Model 4']
x = np.arange(len(models))
 
# Parameter counts per component
D_params     = [6, 6, 6, 6]   # D tensor always 6
Dstar_params = [0, 6, 1, 6]   # D* tensor (6) or scalar (1) or none (0)
f_params     = [0, 1, 6, 6]   # f tensor (6) or scalar (1) or none (0)
 
D_params     = np.array(D_params)
Dstar_params = np.array(Dstar_params)
f_params     = np.array(f_params)
 
total = D_params + Dstar_params + f_params
 
# ── Colors ─────────────────────────────────────────────────────────────────
c_D     = '#185FA5'   # blue  — D tensor
c_Dstar = '#D85A30'   # coral — D*
c_f     = '#1D9E75'   # teal  — f
 
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor('white')
 
width = 0.5
 
# Stacked bars
b1 = ax.bar(x, D_params, width, color=c_D, label='D tensor (6 params)', zorder=3)
b2 = ax.bar(x, Dstar_params, width, bottom=D_params,
            color=c_Dstar, label='D* tensor / scalar', zorder=3)
b3 = ax.bar(x, f_params, width, bottom=D_params + Dstar_params,
            color=c_f, label='f tensor / scalar', zorder=3)
 
# ── Total labels on top of each bar ───────────────────────────────────────
for i, (xi, tot) in enumerate(zip(x, total)):
    ax.text(xi, tot + 0.3, f'{tot} params', ha='center', va='bottom',
            fontsize=10.5, fontweight='bold', color='#2C2C2A')
 
# ── Component labels inside bars ──────────────────────────────────────────
component_labels = [
    ['D tensor\n6', '', ''],
    ['D tensor\n6', 'D* tensor\n6', 'f scalar\n1'],
    ['D tensor\n6', 'D* scalar\n1', 'f tensor\n6'],
    ['D tensor\n6', 'D* tensor\n6', 'f tensor\n6'],
]
 
bottoms = [
    [0, D_params, D_params + Dstar_params],
]
for i in range(4):
    bots = [0, D_params[i], D_params[i] + Dstar_params[i]]
    heights = [D_params[i], Dstar_params[i], f_params[i]]
    for j, (bot, h, lbl) in enumerate(zip(bots, heights, component_labels[i])):
        if h > 0 and lbl:
            ax.text(i, bot + h/2, lbl, ha='center', va='center',
                    fontsize=8.5, color='white', fontweight='bold',
                    linespacing=1.4)
 
# ── Axes ───────────────────────────────────────────────────────────────────
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=11)
ax.set_ylabel('Number of free parameters', fontsize=11)
ax.set_ylim(0, 16)
ax.set_yticks([0, 6, 12, 18])
ax.grid(axis='y', alpha=0.3, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)
 
# ── Legend ─────────────────────────────────────────────────────────────────
legend_patches = [
    mpatches.Patch(color=c_D,     label='D tensor (always 6 params)'),
    mpatches.Patch(color=c_Dstar, label='D* — tensor (6) or scalar (1)'),
    mpatches.Patch(color=c_f,     label='f — tensor (6) or scalar (1)'),
]
ax.legend(handles=legend_patches, fontsize=9.5, loc='upper left',
          framealpha=0.9, edgecolor='lightgray')
 
ax.set_title('Free parameters across IVIM-DTI tensor models',
             fontsize=12, fontweight='bold', pad=12)
 
plt.tight_layout()
plt.savefig('model_complexity.png', dpi=200,
            bbox_inches='tight', facecolor='white')
print("Saved: model_complexity.png")