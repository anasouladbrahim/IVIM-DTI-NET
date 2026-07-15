import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
 
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':   10,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})
 
# ── Colors per fit model ────────────────────────────────────────────────────
colors = {
    'model1': '#185FA5',   # blue
    'model2': '#D85A30',   # coral
    'model3': '#0F6E56',   # green
    'model4': '#854F0B',   # brown
}
 
labels = {
    'model1': 'Model 1',
    'model2': 'Model 2',
    'model3': 'Model 3',
    'model4': 'Model 4',
}
 
# ── Helper: plot one panel, NN solid + LS dashed ────────────────────────────
def plot_panel(ax, datasets, param_key, title, ylabel=True):
    """datasets: list of (label, color, npz_data, linestyle)"""
    for lab, col, data, ls in datasets:
        if f'{param_key}_median' not in data:
            continue
        snr     = data['snr']
        median  = data[f'{param_key}_median']
        iqr_val = data[f'{param_key}_iqr']
        x = np.arange(len(snr))
 
        if ls == '-':
            ax.fill_between(x, median - iqr_val/2, median + iqr_val/2,
                            alpha=0.15, color=col)
        ax.plot(x, median, color=col, linewidth=1.8, linestyle=ls, label=lab)
 
    ax.axhline(0, color='black', linewidth=0.6, alpha=0.4)
    xlabels = [str(int(s)) for s in snr]
    ax.set_xticks(np.arange(len(snr)))
    ax.set_xticklabels(xlabels, fontsize=7, rotation=45)
    ax.set_xlabel('SNR', fontsize=9)
    if ylabel:
        ax.set_ylabel('Relative error (%)', fontsize=9)
    ax.set_title(title, fontweight='bold', fontsize=10)
    ax.grid(True, alpha=0.2, linewidth=0.4)
    ax.legend(fontsize=7, framealpha=0.9)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — Underfitting: Model 4 data → Models 1, 2, 3 (NN solid, LS dashed)
# ══════════════════════════════════════════════════════════════════════════════
print("Making underfitting plot (NN + LS)...")
 
d_m4_m1_nn = np.load('results_mismatch_model4_fitted_model1.npz')
d_m4_m2_nn = np.load('results_mismatch_model4_fitted_model2.npz')
d_m4_m3_nn = np.load('results_mismatch_model4_fitted_model3.npz')
d_m4_m4_nn = np.load('results_nn_model4.npz')   # matched reference
 
d_m4_m1_ls = np.load('results_ls_mismatch_model4_fitted_model1.npz')
d_m4_m2_ls = np.load('results_ls_mismatch_model4_fitted_model2.npz')
d_m4_m3_ls = np.load('results_ls_mismatch_model4_fitted_model3.npz')
 
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
 
all_under = [
    ('Model 1 fit (NN)', colors['model1'], d_m4_m1_nn, '-'),
    ('Model 1 fit (LS)', colors['model1'], d_m4_m1_ls, '--'),
    ('Model 2 fit (NN)', colors['model2'], d_m4_m2_nn, '-'),
    ('Model 2 fit (LS)', colors['model2'], d_m4_m2_ls, '--'),
    ('Model 3 fit (NN)', colors['model3'], d_m4_m3_nn, '-'),
    ('Model 3 fit (LS)', colors['model3'], d_m4_m3_ls, '--'),
]
 
plot_panel(axes1[0,0], all_under, 'RE_MD_diff', r'$\mathrm{MD}_\mathrm{diff}$')
plot_panel(axes1[0,1], all_under, 'RE_FA_diff', r'$\mathrm{FA}_\mathrm{diff}$', ylabel=False)
 
pseudo_under = [
    ('Model 2 fit (NN)', colors['model2'], d_m4_m2_nn, '-'),
    ('Model 2 fit (LS)', colors['model2'], d_m4_m2_ls, '--'),
]
plot_panel(axes1[0,2], pseudo_under, 'RE_MD_pseudo', r'$\mathrm{MD}_\mathrm{pseudo}$', ylabel=False)
 
fa_pseudo_under = [
    ('Model 2 fit (NN)', colors['model2'], d_m4_m2_nn, '-'),
    ('Model 2 fit (LS)', colors['model2'], d_m4_m2_ls, '--'),
]
plot_panel(axes1[1,0], fa_pseudo_under, 'RE_FA_pseudo', r'$\mathrm{FA}_\mathrm{pseudo}$')
 
mf_under = [
    ('Model 3 fit (NN)', colors['model3'], d_m4_m3_nn, '-'),
    ('Model 3 fit (LS)', colors['model3'], d_m4_m3_ls, '--'),
]
plot_panel(axes1[1,1], mf_under, 'RE_Mf', r'$M_f$', ylabel=False)
 
faf_under = [
    ('Model 3 fit (NN)', colors['model3'], d_m4_m3_nn, '-'),
    ('Model 3 fit (LS)', colors['model3'], d_m4_m3_ls, '--'),
]
plot_panel(axes1[1,2], faf_under, 'RE_FA_f', r'$\mathrm{FA}_f$', ylabel=False)
 
fig1.suptitle('Model mismatch — Underfitting (Model 4 data, simpler fit): NN (solid) vs LS (dashed)',
              fontsize=12, fontweight='bold')
plt.savefig('mismatch_underfitting_nn_ls.png', dpi=200,
            bbox_inches='tight', facecolor='white')
print("Saved: mismatch_underfitting_nn_ls.png")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — Overfitting: simpler model data → Model 4 (NN solid, LS dashed)
# ══════════════════════════════════════════════════════════════════════════════
print("Making overfitting plot (NN + LS)...")
 
d_m1_m4_nn = np.load('results_mismatch_model1_fitted_model4.npz')
d_m2_m4_nn = np.load('results_mismatch_model2_fitted_model4.npz')
d_m3_m4_nn = np.load('results_mismatch_model3_fitted_model4.npz')
 
d_m1_m4_ls = np.load('results_ls_mismatch_model1_fitted_model4.npz')
d_m2_m4_ls = np.load('results_ls_mismatch_model2_fitted_model4.npz')
d_m3_m4_ls = np.load('results_ls_mismatch_model3_fitted_model4.npz')
 
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
 
all_over = [
    ('Model 1 data (NN)', colors['model1'], d_m1_m4_nn, '-'),
    ('Model 1 data (LS)', colors['model1'], d_m1_m4_ls, '--'),
    ('Model 2 data (NN)', colors['model2'], d_m2_m4_nn, '-'),
    ('Model 2 data (LS)', colors['model2'], d_m2_m4_ls, '--'),
    ('Model 3 data (NN)', colors['model3'], d_m3_m4_nn, '-'),
    ('Model 3 data (LS)', colors['model3'], d_m3_m4_ls, '--'),
]
 
plot_panel(axes2[0,0], all_over, 'RE_MD_diff', r'$\mathrm{MD}_\mathrm{diff}$')
plot_panel(axes2[0,1], all_over, 'RE_FA_diff', r'$\mathrm{FA}_\mathrm{diff}$', ylabel=False)
 
pseudo_over = [
    ('Model 2 data (NN)', colors['model2'], d_m2_m4_nn, '-'),
    ('Model 2 data (LS)', colors['model2'], d_m2_m4_ls, '--'),
]
plot_panel(axes2[0,2], pseudo_over, 'RE_MD_pseudo', r'$\mathrm{MD}_\mathrm{pseudo}$', ylabel=False)
 
fa_pseudo_over = [
    ('Model 2 data (NN)', colors['model2'], d_m2_m4_nn, '-'),
    ('Model 2 data (LS)', colors['model2'], d_m2_m4_ls, '--'),
]
plot_panel(axes2[1,0], fa_pseudo_over, 'RE_FA_pseudo', r'$\mathrm{FA}_\mathrm{pseudo}$')
 
mf_over = [
    ('Model 3 data (NN)', colors['model3'], d_m3_m4_nn, '-'),
    ('Model 3 data (LS)', colors['model3'], d_m3_m4_ls, '--'),
]
plot_panel(axes2[1,1], mf_over, 'RE_Mf', r'$M_f$', ylabel=False)
 
faf_over = [
    ('Model 3 data (NN)', colors['model3'], d_m3_m4_nn, '-'),
    ('Model 3 data (LS)', colors['model3'], d_m3_m4_ls, '--'),
]
plot_panel(axes2[1,2], faf_over, 'RE_FA_f', r'$\mathrm{FA}_f$', ylabel=False)
 
fig2.suptitle('Model mismatch — Overfitting (Model 4 fit, simpler data): NN (solid) vs LS (dashed)',
              fontsize=12, fontweight='bold')
plt.savefig('mismatch_overfitting_nn_ls.png', dpi=200,
            bbox_inches='tight', facecolor='white')
print("Saved: mismatch_overfitting_nn_ls.png")