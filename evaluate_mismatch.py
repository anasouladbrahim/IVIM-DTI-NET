import numpy as np
import torch
import torch.utils.data as utils
from tqdm import tqdm
from scipy.stats import iqr
import argparse
 
import IVIMDTINET.deep as deep
from hyperparams import hyperparams as hp
 
 
def calc_MD_FA(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz):
    MD = (Dxx + Dyy + Dzz) / 3
    FA = np.sqrt(0.5 * ((Dxx-Dyy)**2 + (Dyy-Dzz)**2 + (Dxx-Dzz)**2 +
                         6*(Dxy**2 + Dxz**2 + Dyz**2))) / \
         np.sqrt(Dxx**2 + Dyy**2 + Dzz**2 + 2*(Dxy**2 + Dxz**2 + Dyz**2) + 1e-8)
    return MD, FA
 
 
# ── Arguments ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--sim_model', type=str, default='model4',
                    choices=['model1', 'model2', 'model3', 'model4'])
parser.add_argument('--fit_model', type=str, default='model1',
                    choices=['model1', 'model2', 'model3', 'model4'])
args = parser.parse_args()
sim_model = args.sim_model
fit_model = args.fit_model
 
print(f"\nMismatch evaluation: simulated with {sim_model}, fitted with {fit_model}")
 
arg = hp()
 
# ── Load bval/bvec ──────────────────────────────────────────────────────────
bval = np.genfromtxt('data/dejong_bval.bval')
bvec = np.genfromtxt('data/dejong_bvec.bvec')
selsb = np.array(bval) == 0
 
# ── Load simulated data ─────────────────────────────────────────────────────
print(f"Loading {sim_model} simulated data...")
signals = np.load(f'data/simulated_{sim_model}_signals.npy', mmap_mode='r')
gt      = np.load(f'data/simulated_{sim_model}_gt.npy', allow_pickle=True)
snr     = np.load(f'data/simulated_{sim_model}_snr.npy')
 
np.random.seed(42)
SNR_list = sorted(np.unique(snr))
idx = []
for s in SNR_list:
    mask = np.where(snr == s)[0]
    idx.extend(np.random.choice(mask, size=min(10000, len(mask)), replace=False))
idx = np.array(idx)
signals = signals[idx].copy()
gt      = gt[idx]
snr     = snr[idx]
 
S0 = np.nanmean(signals[:, selsb], axis=1)
signals = signals / S0[:, None]
 
# ── Load network ────────────────────────────────────────────────────────────
print(f"Loading {fit_model} trained network...")
bval_torch = torch.FloatTensor(bval).to(arg.train_pars.device)
bvec_torch = torch.FloatTensor(bvec).to(arg.train_pars.device)
net = deep.Net(bval_torch, bvec_torch, arg.net_pars, model=fit_model).to(arg.train_pars.device)
net.load_state_dict(torch.load(
    f'trained_networks/{arg.save_name}_simulated_{fit_model}.pt',
    map_location=arg.train_pars.device))
net.eval()
 
# ── Inference ──────────────────────────────────────────────────────────────
inferloader = utils.DataLoader(
    torch.from_numpy(signals.astype(np.float32)),
    batch_size=2056, shuffle=False, drop_last=False)
 
V1 = V2 = V3 = V4 = V5 = V6 = np.array([])
U1 = U2 = U3 = U4 = U5 = U6 = np.array([])
W1 = W2 = W3 = W4 = W5 = W6 = np.array([])
Fp = np.array([])
Dstar_scalar = np.array([])
 
with torch.no_grad():
    for X_batch in tqdm(inferloader):
        X_batch = X_batch.to(arg.train_pars.device)
        outputs = net(X_batch)
 
        if fit_model == 'model1':
            _, V_xxt, V_xyt, V_xzt, V_yyt, V_zzt, V_yzt = outputs
 
        elif fit_model == 'model2':
            _, Fpt, V_xxt, V_xyt, V_xzt, V_yyt, V_zzt, V_yzt, \
            U_xxt, U_xyt, U_xzt, U_yyt, U_zzt, U_yzt, _ = outputs
            Fp = np.append(Fp, Fpt.cpu().numpy())
            U1 = np.append(U1, U_xxt.cpu().numpy())
            U2 = np.append(U2, U_xyt.cpu().numpy())
            U3 = np.append(U3, U_xzt.cpu().numpy())
            U4 = np.append(U4, U_yyt.cpu().numpy())
            U5 = np.append(U5, U_zzt.cpu().numpy())
            U6 = np.append(U6, U_yzt.cpu().numpy())
 
        elif fit_model == 'model3':
            _, Dstar_scalart, V_xxt, V_xyt, V_xzt, V_yyt, V_zzt, V_yzt, \
            W_xxt, W_xyt, W_xzt, W_yyt, W_zzt, W_yzt = outputs
            Dstar_scalar = np.append(Dstar_scalar, Dstar_scalart.cpu().numpy())
            W1 = np.append(W1, W_xxt.cpu().numpy())
            W2 = np.append(W2, W_xyt.cpu().numpy())
            W3 = np.append(W3, W_xzt.cpu().numpy())
            W4 = np.append(W4, W_yyt.cpu().numpy())
            W5 = np.append(W5, W_zzt.cpu().numpy())
            W6 = np.append(W6, W_yzt.cpu().numpy())
 
        elif fit_model == 'model4':
            _, V_xxt, V_xyt, V_xzt, V_yyt, V_zzt, V_yzt, \
            U_xxt, U_xyt, U_xzt, U_yyt, U_zzt, U_yzt, \
            W_xxt, W_xyt, W_xzt, W_yyt, W_zzt, W_yzt = outputs
            U1 = np.append(U1, U_xxt.cpu().numpy())
            U2 = np.append(U2, U_xyt.cpu().numpy())
            U3 = np.append(U3, U_xzt.cpu().numpy())
            U4 = np.append(U4, U_yyt.cpu().numpy())
            U5 = np.append(U5, U_zzt.cpu().numpy())
            U6 = np.append(U6, U_yzt.cpu().numpy())
            W1 = np.append(W1, W_xxt.cpu().numpy())
            W2 = np.append(W2, W_xyt.cpu().numpy())
            W3 = np.append(W3, W_xzt.cpu().numpy())
            W4 = np.append(W4, W_yyt.cpu().numpy())
            W5 = np.append(W5, W_zzt.cpu().numpy())
            W6 = np.append(W6, W_yzt.cpu().numpy())
 
        V1 = np.append(V1, V_xxt.cpu().numpy())
        V2 = np.append(V2, V_xyt.cpu().numpy())
        V3 = np.append(V3, V_xzt.cpu().numpy())
        V4 = np.append(V4, V_yyt.cpu().numpy())
        V5 = np.append(V5, V_zzt.cpu().numpy())
        V6 = np.append(V6, V_yzt.cpu().numpy())
 
# ── Reconstruct D tensor ────────────────────────────────────────────────────
Dxx = V1**2; Dyy = V2**2+V4**2; Dzz = V3**2+V5**2+V6**2
Dxy = V1*V4; Dxz = V1*V5;       Dyz = V4*V5+V2*V6
MD_pred, FA_pred = calc_MD_FA(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)
MD_pred = MD_pred * 1000
 
MD_gt = np.array([g['MD_diff'] for g in gt])
FA_gt = np.array([g['FA_diff'] for g in gt])
RE_MD = (MD_pred - MD_gt) / MD_gt * 100
RE_FA = (FA_pred - FA_gt) / FA_gt * 100
 
save_dict = {'snr': np.array(SNR_list)}
save_dict['RE_MD_diff_median'] = np.array([np.median(RE_MD[snr==s]) for s in SNR_list])
save_dict['RE_MD_diff_iqr']    = np.array([iqr(RE_MD[snr==s])       for s in SNR_list])
save_dict['RE_FA_diff_median'] = np.array([np.median(RE_FA[snr==s]) for s in SNR_list])
save_dict['RE_FA_diff_iqr']    = np.array([iqr(RE_FA[snr==s])       for s in SNR_list])
 
# ── D* tensor — if fit_model has D* tensor AND sim_model has D* ────────────
has_Dstar_tensor_fit = fit_model in ['model2', 'model4']
has_Dstar_sim = sim_model in ['model2', 'model4']
 
if has_Dstar_tensor_fit and has_Dstar_sim:
    Dpxx = U1**2; Dpyy = U2**2+U4**2; Dpzz = U3**2+U5**2+U6**2
    Dpxy = U1*U4; Dpxz = U1*U5;       Dpyz = U4*U5+U2*U6
    MD_pseudo_pred, FA_pseudo_pred = calc_MD_FA(Dpxx, Dpyy, Dpzz, Dpxy, Dpxz, Dpyz)
    MD_pseudo_pred = MD_pseudo_pred * 1000
    MD_pseudo_gt = np.array([g['MD_pseudo'] for g in gt])
    FA_pseudo_gt = np.array([g['FA_pseudo'] for g in gt])
    RE_MDp = (MD_pseudo_pred - MD_pseudo_gt) / MD_pseudo_gt * 100
    RE_FAp = (FA_pseudo_pred - FA_pseudo_gt) / FA_pseudo_gt * 100
    save_dict['RE_MD_pseudo_median'] = np.array([np.median(RE_MDp[snr==s]) for s in SNR_list])
    save_dict['RE_MD_pseudo_iqr']    = np.array([iqr(RE_MDp[snr==s])       for s in SNR_list])
    save_dict['RE_FA_pseudo_median'] = np.array([np.median(RE_FAp[snr==s]) for s in SNR_list])
    save_dict['RE_FA_pseudo_iqr']    = np.array([iqr(RE_FAp[snr==s])       for s in SNR_list])
 
# ── f tensor — if fit_model has f tensor AND sim_model has f tensor ────────
has_f_tensor_fit = fit_model in ['model3', 'model4']
has_f_tensor_sim = sim_model in ['model3', 'model4']
 
if has_f_tensor_fit and has_f_tensor_sim:
    Wfxx = W1**2; Wfyy = W2**2+W4**2; Wfzz = W3**2+W5**2+W6**2
    Wfxy = W1*W4; Wfxz = W1*W5;       Wfyz = W4*W5+W2*W6
    Mf_pred, FAf_pred = calc_MD_FA(Wfxx, Wfyy, Wfzz, Wfxy, Wfxz, Wfyz)
    Mf_gt  = np.array([g['Mf']   for g in gt])
    FAf_gt = np.array([g['FA_f'] for g in gt])
    RE_Mf  = (Mf_pred  - Mf_gt)  / Mf_gt  * 100
    RE_FAf = (FAf_pred - FAf_gt) / FAf_gt * 100
    save_dict['RE_Mf_median']   = np.array([np.median(RE_Mf[snr==s])  for s in SNR_list])
    save_dict['RE_Mf_iqr']      = np.array([iqr(RE_Mf[snr==s])        for s in SNR_list])
    save_dict['RE_FA_f_median'] = np.array([np.median(RE_FAf[snr==s]) for s in SNR_list])
    save_dict['RE_FA_f_iqr']    = np.array([iqr(RE_FAf[snr==s])       for s in SNR_list])
 
# ── Print results ───────────────────────────────────────────────────────────
has_pseudo = 'RE_MD_pseudo_median' in save_dict
has_f      = 'RE_Mf_median' in save_dict
 
header = f'{"SNR":<8} {"MD_diff":<12} {"FA_diff":<12}'
if has_pseudo: header += f'{"MD_pseudo":<12} {"FA_pseudo":<12}'
if has_f:      header += f'{"Mf":<12} {"FA_f":<12}'
print(f'\n{header}')
print('-' * len(header))
 
for s in SNR_list:
    mask = snr == s
    row = f'{int(s):<8} {np.median(RE_MD[mask]):<12.1f} {np.median(RE_FA[mask]):<12.1f}'
    if has_pseudo:
        row += f'{save_dict["RE_MD_pseudo_median"][SNR_list.index(s)]:<12.1f} '
        row += f'{save_dict["RE_FA_pseudo_median"][SNR_list.index(s)]:<12.1f}'
    if has_f:
        row += f'{save_dict["RE_Mf_median"][SNR_list.index(s)]:<12.1f} '
        row += f'{save_dict["RE_FA_f_median"][SNR_list.index(s)]:<12.1f}'
    print(row)
 
# ── Save ────────────────────────────────────────────────────────────────────
fname = f'results_mismatch_{sim_model}_fitted_{fit_model}.npz'
np.savez(fname, **save_dict)
print(f"\nSaved: {fname}")