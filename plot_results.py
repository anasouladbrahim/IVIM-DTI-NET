"""
De Jong-style plots for IVIM-DTI-NET results.
Shows bias (median relative error) + IQR band for NN and LS side by side.
One figure per model, one subplot per parameter.
 
Usage: python3 plot_results.py --model model1
       python3 plot_results.py --model model2
       python3 plot_results.py --model model3
       python3 plot_results.py --model model4
"""
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import argparse
 
# ── Settings ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='model1',
                    choices=['model1', 'model2', 'model3', 'model4'])
args = parser.parse_args()
model = args.model
 
# ── Load results ───────────────────────────────────────────────────────────
nn = np.load(f'results_nn_{model}.npz')
ls = np.load(f'results_ls_{model}.npz')
 
snr = nn['snr']
 
# ── Parameter definitions per model ────────────────────────────────────────
param_configs = {
    'model1': [
        ('RE_MD_diff', r'$\mathrm{MD}_\mathrm{diff}$'),
        ('RE_FA_diff', r'$\mathrm{FA}_\mathrm{diff}$'),
    ],
    'model2': [
        ('RE_MD_diff',   r'$\mathrm{MD}_\mathrm{diff}$'),
        ('RE_FA_diff',   r'$\mathrm{FA}_\mathrm{diff}$'),
        ('RE_f',         r'$f$'),
        ('RE_MD_pseudo', r'$\mathrm{MD}_\mathrm{pseudo}$'),
        ('RE_FA_pseudo', r'$\mathrm{FA}_\mathrm{pseudo}$'),
    ],
    'model3': [
        ('RE_MD_diff', r'$\mathrm{MD}_\mathrm{diff}$'),
        ('RE_FA_diff', r'$\mathrm{FA}_\mathrm{diff}$'),
        ('RE_Mf',      r'$M_f$'),
        ('RE_FA_f',    r'$\mathrm{FA}_f$'),
        ('RE_Dstar',   r'$D^*$'),
    ],
    'model4': [
        ('RE_MD_diff',   r'$\mathrm{MD}_\mathrm{diff}$'),
        ('RE_FA_diff',   r'$\mathrm{FA}_\mathrm{diff}$'),
        ('RE_Mf',        r'$M_f$'),
        ('RE_FA_f',      r'$\mathrm{FA}_f$'),
        ('RE_MD_pseudo', r'$\mathrm{MD}_\mathrm{pseudo}$'),
        ('RE_FA_pseudo', r'$\mathrm{FA}_\mathrm{pseudo}$'),
    ],
}
 
params = param_configs[model]
n_params = len(params)
 
# ── Layout ─────────────────────────────────────────────────────────────────
ncols = 3
nrows = int(np.ceil(n_params / ncols))
 
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
    'xtick.major.width': 0.8,
    'ytick.major.width': 0.8,
})
 
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(5 * ncols, 4 * nrows),
                          constrained_layout=True)
axes = np.array(axes).reshape(-1)
 
# Colors
nn_color = '#D85A30'   # coral — NN
ls_color = '#185FA5'   # blue  — LS
 
# ── Plot each parameter ─────────────────────────────────────────────────────
for i, (key, label) in enumerate(params):
    ax = axes[i]
 
    nn_median = nn[f'{key}_median']
    nn_iqr    = nn[f'{key}_iqr']
    ls_median = ls[f'{key}_median']
    ls_iqr    = ls[f'{key}_iqr']
 
    x = np.arange(len(snr))
    xlabels = [str(int(s)) for s in snr]
 
    # IQR bands
    ax.fill_between(x,
                    nn_median - nn_iqr/2,
                    nn_median + nn_iqr/2,
                    alpha=0.20, color=nn_color, label='_nolegend_')
    ax.fill_between(x,
                    ls_median - ls_iqr/2,
                    ls_median + ls_iqr/2,
                    alpha=0.20, color=ls_color, label='_nolegend_')
 
    # Bias lines
    ax.plot(x, nn_median, color=nn_color, linewidth=1.8,
            label='NN', zorder=3)
    ax.plot(x, ls_median, color=ls_color, linewidth=1.8,
            linestyle='--', label='LS', zorder=3)
 
    # Zero line
    ax.axhline(0, color='black', linewidth=0.6, linestyle='-', alpha=0.4)
 
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45)
    ax.set_xlabel('SNR', fontsize=9)
    ax.set_ylabel('Relative error (%)', fontsize=9)
    ax.set_title(label, fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.2, linewidth=0.4)

    # Clip y-axis for parameters with extreme LS IQR at low SNR
    if key in ['RE_MD_pseudo', 'RE_FA_pseudo']:
        y_min = max(ax.get_ylim()[0], -300)
        y_max = min(ax.get_ylim()[1], 300)
        ax.set_ylim(y_min, y_max)
 
    if i == 0:
        ax.legend(fontsize=9, framealpha=0.9,
                handles=[
                    plt.Line2D([0], [0], color=nn_color, lw=2, label='NN bias'),
                    plt.Line2D([0], [0], color=ls_color, lw=2,
                                linestyle='--', label='LS bias'),
                    plt.Rectangle((0,0), 1, 1, fc=nn_color,
                                    alpha=0.3, label='NN IQR'),
                    plt.Rectangle((0,0), 1, 1, fc=ls_color,
                                    alpha=0.3, label='LS IQR'),
                ])
 
# Hide unused axes
for j in range(n_params, len(axes)):
    axes[j].set_visible(False)
 
model_titles = {
    'model1': 'Model 1 — D tensor only',
    'model2': 'Model 2 — D tensor + D* tensor + f scalar',
    'model3': 'Model 3 — D tensor + D* scalar + f tensor',
    'model4': 'Model 4 — D tensor + D* tensor + f tensor',
}
 
fig.suptitle(f'Cross-model performance: {model_titles[model]}',
             fontsize=12, fontweight='bold')
 
plt.savefig(f'results_{model}.png', dpi=200,
            bbox_inches='tight', facecolor='white')
print(f"Saved: results_{model}.png")