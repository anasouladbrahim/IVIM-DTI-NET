import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import argparse
 
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='model1',
                    choices=['model1', 'model2', 'model3', 'model4'])
args = parser.parse_args()
model = args.model
 
# ── Load ────────────────────────────────────────────────────────────────────
data     = np.load(f'xai_sensitivity_{model}.npz', allow_pickle=True)
bval     = data['bval']
unique_b = data['unique_b']
raw_keys = [k.replace('_raw', '') for k in data.files if k.endswith('_raw')]
 
# ── Build matrix: parameters × b-values (mean over directions) ─────────────
matrix = np.zeros((len(raw_keys), len(unique_b)))
for i, name in enumerate(raw_keys):
    sens = data[f'{name}_raw']
    for j, b in enumerate(unique_b):
        matrix[i, j] = sens[bval == b].mean()
    # normalise per parameter
    matrix[i] /= (matrix[i].max() + 1e-8)
 
# ── Nice parameter labels ───────────────────────────────────────────────────
label_map = {
    'MD_diff':   r'$\mathrm{MD}_\mathrm{diff}$',
    'FA_diff':   r'$\mathrm{FA}_\mathrm{diff}$',
    'f':         r'$f$',
    'MD_pseudo': r'$\mathrm{MD}_\mathrm{pseudo}$',
    'FA_pseudo': r'$\mathrm{FA}_\mathrm{pseudo}$',
    'Mf':        r'$M_f$',
    'FA_f':      r'$\mathrm{FA}_f$',
    'Dstar':     r'$D^*$',
}
ylabels = [label_map.get(k, k) for k in raw_keys]
xlabels = [str(int(b)) for b in unique_b]
 
# ── Plot ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
})
 
fig, ax = plt.subplots(figsize=(10, 0.8 * len(raw_keys) + 2))
fig.patch.set_facecolor('white')
 
im = ax.imshow(matrix, aspect='auto', cmap='YlOrRd',
               vmin=0, vmax=1, interpolation='nearest')
 
# Add value annotations
for i in range(len(raw_keys)):
    for j in range(len(unique_b)):
        val = matrix[i, j]
        color = 'white' if val > 0.6 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=9, color=color, fontweight='bold')
 
# Dashed line at b=200
b200_idx = np.where(unique_b == 200)[0]
if len(b200_idx) > 0:
    ax.axvline(b200_idx[0] - 0.5, color='black',
               linewidth=1.5, linestyle='--', alpha=0.5)
 
ax.set_xticks(range(len(unique_b)))
ax.set_xticklabels(xlabels, fontsize=10)
ax.set_yticks(range(len(raw_keys)))
ax.set_yticklabels(ylabels, fontsize=11)
ax.set_xlabel('b-value (s/mm²)', fontsize=11, labelpad=8)
ax.set_title(f'Gradient sensitivity — {model}',
             fontsize=13, fontweight='bold', pad=12)
 
# Colorbar
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label('Normalised mean |∂θ/∂S|', fontsize=10)
cb.set_ticks([0, 0.5, 1.0])
 
# Clean spines
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
 
plt.tight_layout()
plt.savefig(f'xai_heatmap_clean_{model}.png', dpi=200,
            bbox_inches='tight', facecolor='white')
print(f"Saved: xai_heatmap_clean_{model}.png")