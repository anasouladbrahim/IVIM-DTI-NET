"""
Least-squares fitting for all 4 models on simulated data.
Purpose: validate simulation pipeline and compare against neural network.
At SNR 1000, LS should give ~0% error for all parameters.
 
Following De Jong (2022) and Mozumder (2018).
Usage: python3 least_squares_fitting.py --model model1
"""
 
import numpy as np
from scipy.optimize import least_squares
from tqdm import tqdm
import argparse
 
# ── Arguments ──
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='model1',
                    choices=['model1', 'model2', 'model3', 'model4'])
args = parser.parse_args()
model = args.model
print(f"\nRunning least-squares for {model}...")
 
# ── Load bval/bvec ──
bval = np.genfromtxt('data/dejong_bval.bval')
bvec = np.genfromtxt('data/dejong_bvec.bvec')
selsb = bval == 0
 
# ── Calculate b-matrix ──
gx = bvec[0, :]
gy = bvec[1, :]
gz = bvec[2, :]
bmat = bval[:, None] * np.column_stack([
    gx**2, gy**2, gz**2,
    2*gx*gy, 2*gx*gz, 2*gy*gz
])
 
# direction matrix for f tensor (models 3, 4)
dir_matrix = np.column_stack([
    gx**2, gy**2, gz**2,
    2*gx*gy, 2*gx*gz, 2*gy*gz
])
 
# ── MD and FA calculation ──
def calc_MD_FA(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz):
    MD = (Dxx + Dyy + Dzz) / 3
    num = np.sqrt(0.5 * ((Dxx-Dyy)**2 + (Dyy-Dzz)**2 + (Dxx-Dzz)**2 +
                          6*(Dxy**2 + Dxz**2 + Dyz**2)))
    den = np.sqrt(Dxx**2 + Dyy**2 + Dzz**2 + 2*(Dxy**2 + Dxz**2 + Dyz**2))
    FA = num / (den + 1e-8)
    return MD, FA
 
# ── Cholesky reconstruction (same as evaluate.py) ──
def cholesky_to_tensor(V1, V2, V3, V4, V5, V6):
    Dxx = V1**2
    Dyy = V2**2 + V4**2
    Dzz = V3**2 + V5**2 + V6**2
    Dxy = V1*V4
    Dxz = V1*V6
    Dyz = V2*V5 + V4*V6
    return np.array([Dxx, Dyy, Dzz, Dxy, Dxz, Dyz])
 
# ── Signal models ──
def signal_model1(params, bmat):
    V1, V2, V3, V4, V5, V6 = params
    D = cholesky_to_tensor(V1, V2, V3, V4, V5, V6)
    return np.exp(-bmat @ D)
 
def signal_model2(params, bmat):
    # params = [V1-6 (D), U1-6 (D*), f]
    V1,V2,V3,V4,V5,V6 = params[0:6]
    U1,U2,U3,U4,U5,U6 = params[6:12]
    f = np.abs(params[12])  # abs to keep positive
    D  = cholesky_to_tensor(V1, V2, V3, V4, V5, V6)
    Dp = cholesky_to_tensor(U1, U2, U3, U4, U5, U6)
    return f * np.exp(-bmat @ Dp) + (1-f) * np.exp(-bmat @ D)
 
def signal_model3(params, bmat, bval, dir_matrix):
    # params = [V1-6 (D), W1-6 (f tensor), Dstar_scalar]
    V1,V2,V3,V4,V5,V6 = params[0:6]
    W1,W2,W3,W4,W5,W6 = params[6:12]
    Dstar = np.abs(params[12]) * 1e-3
    D  = cholesky_to_tensor(V1, V2, V3, V4, V5, V6)
    Wf = cholesky_to_tensor(W1, W2, W3, W4, W5, W6)
    gTfg = dir_matrix @ Wf
    gTfg = np.clip(gTfg, 0, 1)
    return gTfg * np.exp(-bval * Dstar) + (1 - gTfg) * np.exp(-bmat @ D)
 
def signal_model4(params, bmat, bval, dir_matrix):
    # params = [V1-6 (D), U1-6 (D*), W1-6 (f tensor)]
    V1,V2,V3,V4,V5,V6 = params[0:6]
    U1,U2,U3,U4,U5,U6 = params[6:12]
    W1,W2,W3,W4,W5,W6 = params[12:18]
    D  = cholesky_to_tensor(V1, V2, V3, V4, V5, V6)
    Dp = cholesky_to_tensor(U1, U2, U3, U4, U5, U6)
    Wf = cholesky_to_tensor(W1, W2, W3, W4, W5, W6)
    gTfg = dir_matrix @ Wf
    gTfg = np.clip(gTfg, 0, 1)
    return gTfg * np.exp(-bmat @ Dp) + (1 - gTfg) * np.exp(-bmat @ D)
 
# ── Residuals ──
def residuals_model1(params, S_meas, bmat):
    return signal_model1(params, bmat) - S_meas
 
def residuals_model2(params, S_meas, bmat):
    return signal_model2(params, bmat) - S_meas
 
def residuals_model3(params, S_meas, bmat, bval, dir_matrix):
    return signal_model3(params, bmat, bval, dir_matrix) - S_meas
 
def residuals_model4(params, S_meas, bmat, bval, dir_matrix):
    return signal_model4(params, bmat, bval, dir_matrix) - S_meas
 
# ── Initial guesses and bounds ──
MD_init    = 1.9e-3   # kidney diffusion midpoint
MDp_init   = 70e-3   # kidney pseudo-diffusion midpoint
Mf_init    = 0.20    # kidney f midpoint
V_init     = np.sqrt(MD_init)
U_init     = np.sqrt(MDp_init)
W_init     = np.sqrt(Mf_init)
Dstar_init = 70.0    # x10^-3 mm^2/s
 
if model == 'model1':
    # [V1, V2, V3, V4, V5, V6]
    x0 = [V_init, 0.0, 0.0, V_init, V_init, 0.0]
    lb = [-9]*6
    ub = [ 9]*6
 
elif model == 'model2':
    # [V1-6, U1-6, f]
    x0 = [V_init, 0.0, 0.0, V_init, V_init, 0.0,
          U_init, 0.0, 0.0, U_init, U_init, 0.0,
          Mf_init]
    lb = [-9]*6 + [-300]*6 + [0.0]
    ub = [ 9]*6 + [ 300]*6 + [1.0]
 
elif model == 'model3':
    # [V1-6, W1-6, Dstar_scalar]
    x0 = [V_init, 0.0, 0.0, V_init, V_init, 0.0,
          W_init, 0.0, 0.0, W_init, W_init, 0.0,
          Dstar_init]
    lb = [-9]*6 + [-9]*6 + [0.0]
    ub = [ 9]*6 + [ 9]*6 + [300.0]
 
elif model == 'model4':
    # [V1-6, U1-6, W1-6]
    x0 = [V_init, 0.0, 0.0, V_init, V_init, 0.0,
          U_init, 0.0, 0.0, U_init, U_init, 0.0,
          W_init, 0.0, 0.0, W_init, W_init, 0.0]
    lb = [-9]*6 + [-300]*6 + [-9]*6
    ub = [ 9]*6 + [ 300]*6 + [ 9]*6
 
# ── Load simulated data ──
print("Loading simulated data...")
gt      = np.load(f'data/simulated_{model}_gt.npy', allow_pickle=True)
signals = np.load(f'data/simulated_{model}_signals.npy', mmap_mode='r')
snr_arr = np.load(f'data/simulated_{model}_snr.npy')
 
# ── Subsample 1000 per SNR ──
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
 
# ── Normalize ──
S0 = np.nanmean(signals_sub[:, selsb], axis=1)
signals_norm = signals_sub / S0[:, None]
 
print(f"Fitting {len(signals_norm)} signals...")
 
# ── Fit each signal ──
n = len(signals_norm)
n_params = len(x0)
params_all = np.zeros((n, n_params))
failed = 0
 
for i in tqdm(range(n)):
    S_meas = signals_norm[i]
    try:
        if model == 'model1':
            result = least_squares(residuals_model1, x0,
                args=(S_meas, bmat), bounds=(lb, ub),
                method='trf', max_nfev=1000)
        elif model == 'model2':
            result = least_squares(residuals_model2, x0,
                args=(S_meas, bmat), bounds=(lb, ub),
                method='trf', max_nfev=1000)
        elif model == 'model3':
            result = least_squares(residuals_model3, x0,
                args=(S_meas, bmat, bval, dir_matrix), bounds=(lb, ub),
                method='trf', max_nfev=1000)
        elif model == 'model4':
            result = least_squares(residuals_model4, x0,
                args=(S_meas, bmat, bval, dir_matrix), bounds=(lb, ub),
                method='trf', max_nfev=1000)
        params_all[i] = result.x
    except Exception:
        params_all[i] = x0
        failed += 1
 
print(f"Failed fits: {failed}/{n}")
 
# ── Extract parameters ──
V1,V2,V3,V4,V5,V6 = params_all[:,0], params_all[:,1], params_all[:,2], \
                     params_all[:,3], params_all[:,4], params_all[:,5]
 
Dxx = V1**2
Dyy = V2**2 + V4**2
Dzz = V3**2 + V5**2 + V6**2
Dxy = V1*V4
Dxz = V1*V6
Dyz = V2*V5 + V4*V6
 
MD_diff_pred, FA_diff_pred = calc_MD_FA(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)
MD_diff_pred = MD_diff_pred * 1000
 
MD_diff_gt = np.array([g['MD_diff'] for g in gt_sub])
FA_diff_gt = np.array([g['FA_diff'] for g in gt_sub])
RE_MD = (MD_diff_pred - MD_diff_gt) / MD_diff_gt * 100
RE_FA = (FA_diff_pred - FA_diff_gt) / FA_diff_gt * 100
 
# ── Model-specific parameters ──
if model == 'model2':
    U1,U2,U3,U4,U5,U6 = params_all[:,6], params_all[:,7], params_all[:,8], \
                         params_all[:,9], params_all[:,10], params_all[:,11]
    Fp = np.abs(params_all[:,12])
    Dpxx = U1**2; Dpyy = U2**2+U4**2; Dpzz = U3**2+U5**2+U6**2
    Dpxy = U1*U4; Dpxz = U1*U6; Dpyz = U2*U5+U4*U6
    MD_pseudo_pred, FA_pseudo_pred = calc_MD_FA(Dpxx, Dpyy, Dpzz, Dpxy, Dpxz, Dpyz)
    MD_pseudo_pred = MD_pseudo_pred * 1000
    Mf_gt        = np.array([g['Mf']        for g in gt_sub])
    MD_pseudo_gt = np.array([g['MD_pseudo'] for g in gt_sub])
    FA_pseudo_gt = np.array([g['FA_pseudo'] for g in gt_sub])
    RE_f         = (Fp - Mf_gt) / Mf_gt * 100
    RE_MD_pseudo = (MD_pseudo_pred - MD_pseudo_gt) / MD_pseudo_gt * 100
    RE_FA_pseudo = (FA_pseudo_pred - FA_pseudo_gt) / FA_pseudo_gt * 100
 
elif model == 'model3':
    W1,W2,W3,W4,W5,W6 = params_all[:,6], params_all[:,7], params_all[:,8], \
                         params_all[:,9], params_all[:,10], params_all[:,11]
    Dstar_scalar = np.abs(params_all[:,12])
    Wfxx = W1**2; Wfyy = W2**2+W4**2; Wfzz = W3**2+W5**2+W6**2
    Wfxy = W1*W4; Wfxz = W1*W6; Wfyz = W2*W5+W4*W6
    MD_f_pred, FA_f_pred = calc_MD_FA(Wfxx, Wfyy, Wfzz, Wfxy, Wfxz, Wfyz)
    Dstar_pred = Dstar_scalar
    Mf_gt        = np.array([g['Mf']        for g in gt_sub])
    FA_f_gt      = np.array([g['FA_f']      for g in gt_sub])
    MD_pseudo_gt = np.array([g['MD_pseudo'] for g in gt_sub])
    RE_Mf    = (MD_f_pred - Mf_gt) / Mf_gt * 100
    RE_FA_f  = (FA_f_pred - FA_f_gt) / FA_f_gt * 100
    RE_Dstar = (Dstar_pred - MD_pseudo_gt) / MD_pseudo_gt * 100
 
elif model == 'model4':
    U1,U2,U3,U4,U5,U6 = params_all[:,6], params_all[:,7], params_all[:,8], \
                         params_all[:,9], params_all[:,10], params_all[:,11]
    W1,W2,W3,W4,W5,W6 = params_all[:,12], params_all[:,13], params_all[:,14], \
                         params_all[:,15], params_all[:,16], params_all[:,17]
    Dpxx = U1**2; Dpyy = U2**2+U4**2; Dpzz = U3**2+U5**2+U6**2
    Dpxy = U1*U4; Dpxz = U1*U6; Dpyz = U2*U5+U4*U6
    MD_pseudo_pred, FA_pseudo_pred = calc_MD_FA(Dpxx, Dpyy, Dpzz, Dpxy, Dpxz, Dpyz)
    MD_pseudo_pred = MD_pseudo_pred * 1000
    Wfxx = W1**2; Wfyy = W2**2+W4**2; Wfzz = W3**2+W5**2+W6**2
    Wfxy = W1*W4; Wfxz = W1*W6; Wfyz = W2*W5+W4*W6
    MD_f_pred, FA_f_pred = calc_MD_FA(Wfxx, Wfyy, Wfzz, Wfxy, Wfxz, Wfyz)
    Mf_gt        = np.array([g['Mf']        for g in gt_sub])
    FA_f_gt      = np.array([g['FA_f']      for g in gt_sub])
    MD_pseudo_gt = np.array([g['MD_pseudo'] for g in gt_sub])
    FA_pseudo_gt = np.array([g['FA_pseudo'] for g in gt_sub])
    RE_Mf        = (MD_f_pred - Mf_gt) / Mf_gt * 100
    RE_FA_f      = (FA_f_pred - FA_f_gt) / FA_f_gt * 100
    RE_MD_pseudo = (MD_pseudo_pred - MD_pseudo_gt) / MD_pseudo_gt * 100
    RE_FA_pseudo = (FA_pseudo_pred - FA_pseudo_gt) / FA_pseudo_gt * 100
 
# ── Print results ──
if model == 'model1':
    print(f"\n{'SNR':<8} {'MD_diff':<12} {'FA_diff':<12}")
    print('-' * 32)
    for s in SNR_list:
        mask = snr_sub == s
        print(f"{int(s):<8} "
              f"{np.median(RE_MD[mask]):<12.1f} "
              f"{np.median(RE_FA[mask]):<12.1f}")
 
elif model == 'model2':
    print(f"\n{'SNR':<8} {'MD_diff':<12} {'FA_diff':<12} {'f':<12} {'MD_pseudo':<12} {'FA_pseudo':<12}")
    print('-' * 68)
    for s in SNR_list:
        mask = snr_sub == s
        print(f"{int(s):<8} "
              f"{np.median(RE_MD[mask]):<12.1f} "
              f"{np.median(RE_FA[mask]):<12.1f} "
              f"{np.median(RE_f[mask]):<12.1f} "
              f"{np.median(RE_MD_pseudo[mask]):<12.1f} "
              f"{np.median(RE_FA_pseudo[mask]):<12.1f}")
 
elif model == 'model3':
    print(f"\n{'SNR':<8} {'MD_diff':<12} {'FA_diff':<12} {'Mf':<12} {'FA_f':<12} {'Dstar':<12}")
    print('-' * 68)
    for s in SNR_list:
        mask = snr_sub == s
        print(f"{int(s):<8} "
              f"{np.median(RE_MD[mask]):<12.1f} "
              f"{np.median(RE_FA[mask]):<12.1f} "
              f"{np.median(RE_Mf[mask]):<12.1f} "
              f"{np.median(RE_FA_f[mask]):<12.1f} "
              f"{np.median(RE_Dstar[mask]):<12.1f}")
 
elif model == 'model4':
    print(f"\n{'SNR':<8} {'MD_diff':<12} {'FA_diff':<12} {'Mf':<12} {'FA_f':<12} {'MD_pseudo':<12} {'FA_pseudo':<12}")
    print('-' * 80)
    for s in SNR_list:
        mask = snr_sub == s
        print(f"{int(s):<8} "
              f"{np.median(RE_MD[mask]):<12.1f} "
              f"{np.median(RE_FA[mask]):<12.1f} "
              f"{np.median(RE_Mf[mask]):<12.1f} "
              f"{np.median(RE_FA_f[mask]):<12.1f} "
              f"{np.median(RE_MD_pseudo[mask]):<12.1f} "
              f"{np.median(RE_FA_pseudo[mask]):<12.1f}")
        


# ── Save results for plotting ──────────────────────────────────────────────
from scipy.stats import iqr
SNR_arr  = np.array(SNR_list)
save_dict = {'snr': SNR_arr}

save_dict['RE_MD_diff_median'] = np.array([np.median(RE_MD[snr_sub==s]) for s in SNR_list])
save_dict['RE_MD_diff_iqr']    = np.array([iqr(RE_MD[snr_sub==s])       for s in SNR_list])
save_dict['RE_FA_diff_median'] = np.array([np.median(RE_FA[snr_sub==s]) for s in SNR_list])
save_dict['RE_FA_diff_iqr']    = np.array([iqr(RE_FA[snr_sub==s])       for s in SNR_list])

if model == 'model2':
    save_dict['RE_f_median']         = np.array([np.median(RE_f[snr_sub==s])         for s in SNR_list])
    save_dict['RE_f_iqr']            = np.array([iqr(RE_f[snr_sub==s])               for s in SNR_list])
    save_dict['RE_MD_pseudo_median'] = np.array([np.median(RE_MD_pseudo[snr_sub==s]) for s in SNR_list])
    save_dict['RE_MD_pseudo_iqr']    = np.array([iqr(RE_MD_pseudo[snr_sub==s])       for s in SNR_list])
    save_dict['RE_FA_pseudo_median'] = np.array([np.median(RE_FA_pseudo[snr_sub==s]) for s in SNR_list])
    save_dict['RE_FA_pseudo_iqr']    = np.array([iqr(RE_FA_pseudo[snr_sub==s])       for s in SNR_list])

elif model == 'model3':
    save_dict['RE_Mf_median']    = np.array([np.median(RE_Mf[snr_sub==s])    for s in SNR_list])
    save_dict['RE_Mf_iqr']       = np.array([iqr(RE_Mf[snr_sub==s])          for s in SNR_list])
    save_dict['RE_FA_f_median']  = np.array([np.median(RE_FA_f[snr_sub==s])  for s in SNR_list])
    save_dict['RE_FA_f_iqr']     = np.array([iqr(RE_FA_f[snr_sub==s])        for s in SNR_list])
    save_dict['RE_Dstar_median'] = np.array([np.median(RE_Dstar[snr_sub==s]) for s in SNR_list])
    save_dict['RE_Dstar_iqr']    = np.array([iqr(RE_Dstar[snr_sub==s])       for s in SNR_list])

elif model == 'model4':
    save_dict['RE_Mf_median']        = np.array([np.median(RE_Mf[snr_sub==s])        for s in SNR_list])
    save_dict['RE_Mf_iqr']           = np.array([iqr(RE_Mf[snr_sub==s])              for s in SNR_list])
    save_dict['RE_FA_f_median']      = np.array([np.median(RE_FA_f[snr_sub==s])      for s in SNR_list])
    save_dict['RE_FA_f_iqr']         = np.array([iqr(RE_FA_f[snr_sub==s])            for s in SNR_list])
    save_dict['RE_MD_pseudo_median'] = np.array([np.median(RE_MD_pseudo[snr_sub==s]) for s in SNR_list])
    save_dict['RE_MD_pseudo_iqr']    = np.array([iqr(RE_MD_pseudo[snr_sub==s])       for s in SNR_list])
    save_dict['RE_FA_pseudo_median'] = np.array([np.median(RE_FA_pseudo[snr_sub==s]) for s in SNR_list])
    save_dict['RE_FA_pseudo_iqr']    = np.array([iqr(RE_FA_pseudo[snr_sub==s])       for s in SNR_list])

np.savez(f'results_ls_{model}.npz', **save_dict)
print(f"\nSaved: results_ls_{model}.npz")