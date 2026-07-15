"""
Least-squares model mismatch evaluation.
Fits simulated data from one model using the signal equation of another model.
 
Usage:
    python3 least_squares_mismatch.py --sim_model model4 --fit_model model1
    python3 least_squares_mismatch.py --sim_model model1 --fit_model model4
"""
 
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import iqr
from tqdm import tqdm
import argparse
 
# ── Arguments ──
parser = argparse.ArgumentParser()
parser.add_argument('--sim_model', type=str, default='model4',
                    choices=['model1', 'model2', 'model3', 'model4'])
parser.add_argument('--fit_model', type=str, default='model1',
                    choices=['model1', 'model2', 'model3', 'model4'])
args = parser.parse_args()
sim_model = args.sim_model
fit_model = args.fit_model
print(f"\nLS mismatch: simulated with {sim_model}, fitted with {fit_model}")
 
# ── Load bval/bvec ──
bval = np.genfromtxt('data/dejong_bval.bval')
bvec = np.genfromtxt('data/dejong_bvec.bvec')
selsb = bval == 0
 
gx, gy, gz = bvec[0, :], bvec[1, :], bvec[2, :]
bmat = bval[:, None] * np.column_stack([
    gx**2, gy**2, gz**2, 2*gx*gy, 2*gx*gz, 2*gy*gz
])
dir_matrix = np.column_stack([
    gx**2, gy**2, gz**2, 2*gx*gy, 2*gx*gz, 2*gy*gz
])
 
def calc_MD_FA(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz):
    MD = (Dxx + Dyy + Dzz) / 3
    num = np.sqrt(0.5 * ((Dxx-Dyy)**2 + (Dyy-Dzz)**2 + (Dxx-Dzz)**2 +
                          6*(Dxy**2 + Dxz**2 + Dyz**2)))
    den = np.sqrt(Dxx**2 + Dyy**2 + Dzz**2 + 2*(Dxy**2 + Dxz**2 + Dyz**2))
    return MD, num / (den + 1e-8)
 
def cholesky_to_tensor(V1, V2, V3, V4, V5, V6):
    Dxx = V1**2
    Dyy = V2**2 + V4**2
    Dzz = V3**2 + V5**2 + V6**2
    Dxy = V1*V4
    Dxz = V1*V6
    Dyz = V2*V5 + V4*V6
    return np.array([Dxx, Dyy, Dzz, Dxy, Dxz, Dyz])
 
# ── Signal models (identical to least_squares_fitting.py) ──
def signal_model1(params, bmat):
    D = cholesky_to_tensor(*params[0:6])
    return np.exp(-bmat @ D)
 
def signal_model2(params, bmat):
    D  = cholesky_to_tensor(*params[0:6])
    Dp = cholesky_to_tensor(*params[6:12])
    f  = np.abs(params[12])
    return f * np.exp(-bmat @ Dp) + (1-f) * np.exp(-bmat @ D)
 
def signal_model3(params, bmat, bval, dir_matrix):
    D  = cholesky_to_tensor(*params[0:6])
    Wf = cholesky_to_tensor(*params[6:12])
    Dstar = np.abs(params[12]) * 1e-3
    gTfg = np.clip(dir_matrix @ Wf, 0, 1)
    return gTfg * np.exp(-bval * Dstar) + (1 - gTfg) * np.exp(-bmat @ D)
 
def signal_model4(params, bmat, bval, dir_matrix):
    D  = cholesky_to_tensor(*params[0:6])
    Dp = cholesky_to_tensor(*params[6:12])
    Wf = cholesky_to_tensor(*params[12:18])
    gTfg = np.clip(dir_matrix @ Wf, 0, 1)
    return gTfg * np.exp(-bmat @ Dp) + (1 - gTfg) * np.exp(-bmat @ D)
 
def residuals(params, S_meas, fit_model, bmat, bval, dir_matrix):
    if fit_model == 'model1':
        return signal_model1(params, bmat) - S_meas
    elif fit_model == 'model2':
        return signal_model2(params, bmat) - S_meas
    elif fit_model == 'model3':
        return signal_model3(params, bmat, bval, dir_matrix) - S_meas
    elif fit_model == 'model4':
        return signal_model4(params, bmat, bval, dir_matrix) - S_meas
 
# ── Initial guesses and bounds (depend on FIT model, not sim model) ──
MD_init, MDp_init, Mf_init = 1.9e-3, 70e-3, 0.20
V_init, U_init, W_init = np.sqrt(MD_init), np.sqrt(MDp_init), np.sqrt(Mf_init)
Dstar_init = 70.0
 
if fit_model == 'model1':
    x0 = [V_init, 0.0, 0.0, V_init, V_init, 0.0]
    lb = [-9]*6
    ub = [ 9]*6
elif fit_model == 'model2':
    x0 = [V_init, 0.0, 0.0, V_init, V_init, 0.0,
          U_init, 0.0, 0.0, U_init, U_init, 0.0, Mf_init]
    lb = [-9]*6 + [-300]*6 + [0.0]
    ub = [ 9]*6 + [ 300]*6 + [1.0]
elif fit_model == 'model3':
    x0 = [V_init, 0.0, 0.0, V_init, V_init, 0.0,
          W_init, 0.0, 0.0, W_init, W_init, 0.0, Dstar_init]
    lb = [-9]*6 + [-9]*6 + [0.0]
    ub = [ 9]*6 + [ 9]*6 + [300.0]
elif fit_model == 'model4':
    x0 = [V_init, 0.0, 0.0, V_init, V_init, 0.0,
          U_init, 0.0, 0.0, U_init, U_init, 0.0,
          W_init, 0.0, 0.0, W_init, W_init, 0.0]
    lb = [-9]*6 + [-300]*6 + [-9]*6
    ub = [ 9]*6 + [ 300]*6 + [ 9]*6
 
# ── Load SIMULATED data (from sim_model) ──
print(f"Loading {sim_model} simulated data...")
gt      = np.load(f'data/simulated_{sim_model}_gt.npy', allow_pickle=True)
signals = np.load(f'data/simulated_{sim_model}_signals.npy', mmap_mode='r')
snr_arr = np.load(f'data/simulated_{sim_model}_snr.npy')
 
SNR_list = list(range(5, 65, 5)) + [1000]
np.random.seed(42)
idx = []
for s in SNR_list:
    mask = np.where(snr_arr == s)[0]
    idx.extend(np.random.choice(mask, size=min(1000, len(mask)), replace=False))
idx = np.array(idx)
 
signals_sub = signals[idx].copy()
gt_sub      = gt[idx]
snr_sub     = snr_arr[idx]
 
S0 = np.nanmean(signals_sub[:, selsb], axis=1)
signals_norm = signals_sub / S0[:, None]
 
print(f"Fitting {len(signals_norm)} signals with {fit_model} signal equation...")
 
n = len(signals_norm)
params_all = np.zeros((n, len(x0)))
failed = 0
 
for i in tqdm(range(n)):
    S_meas = signals_norm[i]
    try:
        result = least_squares(residuals, x0,
            args=(S_meas, fit_model, bmat, bval, dir_matrix),
            bounds=(lb, ub), method='trf', max_nfev=1000)
        params_all[i] = result.x
    except Exception:
        params_all[i] = x0
        failed += 1
 
print(f"Failed fits: {failed}/{n}")
 
# ── Extract D-tensor (always present, evaluated against sim_model ground truth) ──
V1,V2,V3,V4,V5,V6 = [params_all[:,k] for k in range(6)]
Dxx = V1**2; Dyy = V2**2+V4**2; Dzz = V3**2+V5**2+V6**2
Dxy = V1*V4; Dxz = V1*V6;       Dyz = V2*V5+V4*V6
MD_pred, FA_pred = calc_MD_FA(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)
MD_pred = MD_pred * 1000
 
MD_gt = np.array([g['MD_diff'] for g in gt_sub])
FA_gt = np.array([g['FA_diff'] for g in gt_sub])
RE_MD = (MD_pred - MD_gt) / MD_gt * 100
RE_FA = (FA_pred - FA_gt) / FA_gt * 100
 
save_dict = {'snr': np.array(SNR_list)}
save_dict['RE_MD_diff_median'] = np.array([np.median(RE_MD[snr_sub==s]) for s in SNR_list])
save_dict['RE_MD_diff_iqr']    = np.array([iqr(RE_MD[snr_sub==s])       for s in SNR_list])
save_dict['RE_FA_diff_median'] = np.array([np.median(RE_FA[snr_sub==s]) for s in SNR_list])
save_dict['RE_FA_diff_iqr']    = np.array([iqr(RE_FA[snr_sub==s])       for s in SNR_list])
 
# ── D* tensor — only if BOTH sim and fit model have it (models 2, 4) ──
has_Dstar_fit = fit_model in ['model2', 'model4']
has_Dstar_sim = sim_model in ['model2', 'model4']
if has_Dstar_fit and has_Dstar_sim:
    U1,U2,U3,U4,U5,U6 = [params_all[:,k] for k in range(6,12)]
    Dpxx = U1**2; Dpyy = U2**2+U4**2; Dpzz = U3**2+U5**2+U6**2
    Dpxy = U1*U4; Dpxz = U1*U6;       Dpyz = U2*U5+U4*U6
    MD_pseudo_pred, FA_pseudo_pred = calc_MD_FA(Dpxx, Dpyy, Dpzz, Dpxy, Dpxz, Dpyz)
    MD_pseudo_pred = MD_pseudo_pred * 1000
    MD_pseudo_gt = np.array([g['MD_pseudo'] for g in gt_sub])
    FA_pseudo_gt = np.array([g['FA_pseudo'] for g in gt_sub])
    RE_MDp = (MD_pseudo_pred - MD_pseudo_gt) / MD_pseudo_gt * 100
    RE_FAp = (FA_pseudo_pred - FA_pseudo_gt) / FA_pseudo_gt * 100
    save_dict['RE_MD_pseudo_median'] = np.array([np.median(RE_MDp[snr_sub==s]) for s in SNR_list])
    save_dict['RE_MD_pseudo_iqr']    = np.array([iqr(RE_MDp[snr_sub==s])       for s in SNR_list])
    save_dict['RE_FA_pseudo_median'] = np.array([np.median(RE_FAp[snr_sub==s]) for s in SNR_list])
    save_dict['RE_FA_pseudo_iqr']    = np.array([iqr(RE_FAp[snr_sub==s])       for s in SNR_list])
 
# ── f tensor — only if BOTH sim and fit model have it (models 3, 4) ──
has_f_fit = fit_model in ['model3', 'model4']
has_f_sim = sim_model in ['model3', 'model4']
if has_f_fit and has_f_sim:
    w_idx = 12 if fit_model == 'model4' else 6
    W1,W2,W3,W4,W5,W6 = [params_all[:,k] for k in range(w_idx, w_idx+6)]
    Wfxx = W1**2; Wfyy = W2**2+W4**2; Wfzz = W3**2+W5**2+W6**2
    Wfxy = W1*W4; Wfxz = W1*W6;       Wfyz = W2*W5+W4*W6
    Mf_pred, FAf_pred = calc_MD_FA(Wfxx, Wfyy, Wfzz, Wfxy, Wfxz, Wfyz)
    Mf_gt   = np.array([g['Mf']   for g in gt_sub])
    FAf_gt  = np.array([g['FA_f'] for g in gt_sub])
    RE_Mf   = (Mf_pred  - Mf_gt)  / Mf_gt  * 100
    RE_FAf  = (FAf_pred - FAf_gt) / FAf_gt * 100
    save_dict['RE_Mf_median']   = np.array([np.median(RE_Mf[snr_sub==s])  for s in SNR_list])
    save_dict['RE_Mf_iqr']      = np.array([iqr(RE_Mf[snr_sub==s])        for s in SNR_list])
    save_dict['RE_FA_f_median'] = np.array([np.median(RE_FAf[snr_sub==s]) for s in SNR_list])
    save_dict['RE_FA_f_iqr']    = np.array([iqr(RE_FAf[snr_sub==s])       for s in SNR_list])
 
# ── Print results ──
has_pseudo = 'RE_MD_pseudo_median' in save_dict
has_f      = 'RE_Mf_median' in save_dict
header = f'{"SNR":<8} {"MD_diff":<12} {"FA_diff":<12}'
if has_pseudo: header += f'{"MD_pseudo":<12} {"FA_pseudo":<12}'
if has_f:      header += f'{"Mf":<12} {"FA_f":<12}'
print(f'\n{header}')
print('-' * len(header))
for j, s in enumerate(SNR_list):
    row = f'{int(s):<8} {save_dict["RE_MD_diff_median"][j]:<12.1f} {save_dict["RE_FA_diff_median"][j]:<12.1f}'
    if has_pseudo:
        row += f'{save_dict["RE_MD_pseudo_median"][j]:<12.1f} {save_dict["RE_FA_pseudo_median"][j]:<12.1f}'
    if has_f:
        row += f'{save_dict["RE_Mf_median"][j]:<12.1f} {save_dict["RE_FA_f_median"][j]:<12.1f}'
    print(row)
 
fname = f'results_ls_mismatch_{sim_model}_fitted_{fit_model}.npz'
np.savez(fname, **save_dict)
print(f"\nSaved: {fname}")