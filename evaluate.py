#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation script for IVIM-DTI-NET on simulated data.
Calculates relative error between network predictions and ground truth.
Following De Jong (2022).
"""
import numpy as np
import torch
import torch.utils.data as utils
from tqdm import tqdm
import IVIMDTINET.deep as deep
from hyperparams import hyperparams as hp
import argparse


def calc_MD_FA(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz):
    """Calculate MD and FA from tensor elements."""
    MD = (Dxx + Dyy + Dzz) / 3
    FA = np.sqrt(0.5 * ((Dxx - Dyy)**2 + (Dyy - Dzz)**2 + (Dxx - Dzz)**2 +
                         6 * (Dxy**2 + Dxz**2 + Dyz**2))) / \
         np.sqrt(Dxx**2 + Dyy**2 + Dzz**2 + 2 * (Dxy**2 + Dxz**2 + Dyz**2))
    return MD, FA

# Settings 
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='model2',
                    choices=['model1', 'model2', 'model3', 'model4'])
args_parsed = parser.parse_args()
model = args_parsed.model

arg = hp()

# Load bval/bvec 
bval = np.genfromtxt('data/dejong_bval.bval')
bvec = np.genfromtxt('data/dejong_bvec.bvec')
selsb = np.array(bval) == 0

# Load simulated data
signals = np.load(f'data/simulated_{model}_signals.npy')
gt      = np.load(f'data/simulated_{model}_gt.npy', allow_pickle=True)
snr     = np.load(f'data/simulated_{model}_snr.npy')

# Normalize 
S0 = np.nanmean(signals[:, selsb], axis=1)
signals = signals / S0[:, None]

# Load trained network 
bval_torch = torch.FloatTensor(bval).to(arg.train_pars.device)
bvec_torch = torch.FloatTensor(bvec).to(arg.train_pars.device)
net = deep.Net(bval_torch, bvec_torch, arg.net_pars, model=model).to(arg.train_pars.device)
net.load_state_dict(torch.load(
    f'trained_networks/{arg.save_name}_simulated_{model}.pt',
    map_location=arg.train_pars.device))
net.eval()

# Prediction
inferloader = utils.DataLoader(
    torch.from_numpy(signals.astype(np.float32)),
    batch_size=2056,
    shuffle=False,
    drop_last=False)

V1 = V2 = V3 = V4 = V5 = V6 = np.array([])
U1 = U2 = U3 = U4 = U5 = U6 = np.array([])
Fp = np.array([])
W1 = W2 = W3 = W4 = W5 = W6 = np.array([])
Dstar_scalar = np.array([])

with torch.no_grad():
    for X_batch in tqdm(inferloader):
        X_batch = X_batch.to(arg.train_pars.device)
        if model == 'model1':
            _, V_xxt, V_xyt, V_xzt, V_yyt, V_zzt, V_yzt = net(X_batch)
        elif model == 'model2':
            _, Fpt, V_xxt, V_xyt, V_xzt, V_yyt, V_zzt, V_yzt, \
            U_xxt, U_xyt, U_xzt, U_yyt, U_zzt, U_yzt, _ = net(X_batch)
            Fp = np.append(Fp, Fpt.cpu().numpy())
            U1 = np.append(U1, U_xxt.cpu().numpy())
            U2 = np.append(U2, U_xyt.cpu().numpy())
            U3 = np.append(U3, U_xzt.cpu().numpy())
            U4 = np.append(U4, U_yyt.cpu().numpy())
            U5 = np.append(U5, U_yzt.cpu().numpy())
            U6 = np.append(U6, U_zzt.cpu().numpy())
        elif model == 'model3':
            _, Dstar_scalart, V_xxt, V_xyt, V_xzt, V_yyt, V_zzt, V_yzt, \
            W_xxt, W_xyt, W_xzt, W_yyt, W_zzt, W_yzt = net(X_batch)
            Dstar_scalar = np.append(Dstar_scalar, Dstar_scalart.cpu().numpy())
            W1 = np.append(W1, W_xxt.cpu().numpy())
            W2 = np.append(W2, W_xyt.cpu().numpy())
            W3 = np.append(W3, W_xzt.cpu().numpy())
            W4 = np.append(W4, W_yyt.cpu().numpy())
            W5 = np.append(W5, W_yzt.cpu().numpy())
            W6 = np.append(W6, W_zzt.cpu().numpy())
        elif model == 'model4':
            _, V_xxt, V_xyt, V_xzt, V_yyt, V_zzt, V_yzt, \
            U_xxt, U_xyt, U_xzt, U_yyt, U_zzt, U_yzt, \
            W_xxt, W_xyt, W_xzt, W_yyt, W_zzt, W_yzt = net(X_batch)
            U1 = np.append(U1, U_xxt.cpu().numpy())
            U2 = np.append(U2, U_xyt.cpu().numpy())
            U3 = np.append(U3, U_xzt.cpu().numpy())
            U4 = np.append(U4, U_yyt.cpu().numpy())
            U5 = np.append(U5, U_yzt.cpu().numpy())
            U6 = np.append(U6, U_zzt.cpu().numpy())
            W1 = np.append(W1, W_xxt.cpu().numpy())
            W2 = np.append(W2, W_xyt.cpu().numpy())
            W3 = np.append(W3, W_xzt.cpu().numpy())
            W4 = np.append(W4, W_yyt.cpu().numpy())
            W5 = np.append(W5, W_yzt.cpu().numpy())
            W6 = np.append(W6, W_zzt.cpu().numpy())
        V1 = np.append(V1, V_xxt.cpu().numpy())
        V2 = np.append(V2, V_xyt.cpu().numpy())
        V3 = np.append(V3, V_xzt.cpu().numpy())
        V4 = np.append(V4, V_yyt.cpu().numpy())
        V5 = np.append(V5, V_yzt.cpu().numpy())
        V6 = np.append(V6, V_zzt.cpu().numpy())

if model == 'model3':
    print(f"Dstar_scalar range: {Dstar_scalar.min():.4f} to {Dstar_scalar.max():.4f}")
    print(f"Dstar_scalar mean: {Dstar_scalar.mean():.4f}")

# ── Calculate tensor elements from Cholesky ──
Dxx = V1**2
Dyy = V2**2 + V4**2
Dzz = V3**2 + V5**2 + V6**2
Dxy = V1*V4
Dyz = V2*V5 + V4*V6
Dxz = V1*V6

MD_diff_pred, FA_diff_pred = calc_MD_FA(Dxx, Dyy, Dzz, Dxy, Dxz, Dyz)
MD_diff_pred = MD_diff_pred * 1000

MD_diff_gt = np.array([g['MD_diff'] for g in gt])
FA_diff_gt = np.array([g['FA_diff'] for g in gt])
RE_MD_diff = (MD_diff_pred - MD_diff_gt) / MD_diff_gt * 100
RE_FA_diff = (FA_diff_pred - FA_diff_gt) / FA_diff_gt * 100

if model == 'model2':
    Dpxx = U1**2
    Dpyy = U2**2 + U4**2
    Dpzz = U3**2 + U5**2 + U6**2
    Dpxy = U1*U4
    Dpyz = U2*U5 + U4*U6
    Dpxz = U1*U6
    MD_pseudo_pred, FA_pseudo_pred = calc_MD_FA(Dpxx, Dpyy, Dpzz, Dpxy, Dpxz, Dpyz)
    MD_pseudo_pred = MD_pseudo_pred * 1000
    Mf_gt        = np.array([g['Mf']        for g in gt])
    MD_pseudo_gt = np.array([g['MD_pseudo'] for g in gt])
    FA_pseudo_gt = np.array([g['FA_pseudo'] for g in gt])
    RE_f         = (Fp - Mf_gt) / Mf_gt * 100
    RE_MD_pseudo = (MD_pseudo_pred - MD_pseudo_gt) / MD_pseudo_gt * 100
    RE_FA_pseudo = (FA_pseudo_pred - FA_pseudo_gt) / FA_pseudo_gt * 100
elif model == 'model3':
    Wfxx = W1**2
    Wfyy = W2**2 + W4**2
    Wfzz = W3**2 + W5**2 + W6**2
    Wfxy = W1*W4
    Wfyz = W2*W5 + W4*W6
    Wfxz = W1*W6
    MD_f_pred, FA_f_pred = calc_MD_FA(Wfxx, Wfyy, Wfzz, Wfxy, Wfxz, Wfyz)
    Dstar_scalar_pred = Dstar_scalar * 1000
    Mf_gt        = np.array([g['Mf']        for g in gt])
    FA_f_gt      = np.array([g['FA_f']      for g in gt])
    MD_pseudo_gt = np.array([g['MD_pseudo'] for g in gt])
    RE_Mf        = (MD_f_pred - Mf_gt) / Mf_gt * 100
    RE_FA_f      = (FA_f_pred - FA_f_gt) / FA_f_gt * 100
    RE_Dstar     = (Dstar_scalar_pred - MD_pseudo_gt) / MD_pseudo_gt * 100
elif model == 'model4':
    Dpxx = U1**2
    Dpyy = U2**2 + U4**2
    Dpzz = U3**2 + U5**2 + U6**2
    Dpxy = U1*U4
    Dpyz = U2*U5 + U4*U6
    Dpxz = U1*U6
    MD_pseudo_pred, FA_pseudo_pred = calc_MD_FA(Dpxx, Dpyy, Dpzz, Dpxy, Dpxz, Dpyz)
    MD_pseudo_pred = MD_pseudo_pred * 1000
    Wfxx = W1**2
    Wfyy = W2**2 + W4**2
    Wfzz = W3**2 + W5**2 + W6**2
    Wfxy = W1*W4
    Wfyz = W2*W5 + W4*W6
    Wfxz = W1*W6
    MD_f_pred, FA_f_pred = calc_MD_FA(Wfxx, Wfyy, Wfzz, Wfxy, Wfxz, Wfyz)
    Mf_gt        = np.array([g['Mf']        for g in gt])
    FA_f_gt      = np.array([g['FA_f']      for g in gt])
    MD_pseudo_gt = np.array([g['MD_pseudo'] for g in gt])
    FA_pseudo_gt = np.array([g['FA_pseudo'] for g in gt])
    RE_Mf        = (MD_f_pred - Mf_gt) / Mf_gt * 100
    RE_FA_f      = (FA_f_pred - FA_f_gt) / FA_f_gt * 100
    RE_MD_pseudo = (MD_pseudo_pred - MD_pseudo_gt) / MD_pseudo_gt * 100
    RE_FA_pseudo = (FA_pseudo_pred - FA_pseudo_gt) / FA_pseudo_gt * 100

SNR_list = sorted(np.unique(snr))
if model == 'model1':
    print(f'\n{"SNR":<8} {"MD_diff":<12} {"FA_diff":<12}')
    print('-' * 32)
    for s in SNR_list:
        mask = snr == s
        print(f'{int(s):<8} '
              f'{np.median(RE_MD_diff[mask]):<12.1f} '
              f'{np.median(RE_FA_diff[mask]):<12.1f}')
elif model == 'model2':
    print(f'\n{"SNR":<8} {"MD_diff":<12} {"FA_diff":<12} {"f":<12} {"MD_pseudo":<12} {"FA_pseudo":<12}')
    print('-' * 68)
    for s in SNR_list:
        mask = snr == s
        print(f'{int(s):<8} '
              f'{np.median(RE_MD_diff[mask]):<12.1f} '
              f'{np.median(RE_FA_diff[mask]):<12.1f} '
              f'{np.median(RE_f[mask]):<12.1f} '
              f'{np.median(RE_MD_pseudo[mask]):<12.1f} '
              f'{np.median(RE_FA_pseudo[mask]):<12.1f}')
elif model == 'model3':
    print(f'\n{"SNR":<8} {"MD_diff":<12} {"FA_diff":<12} {"Mf":<12} {"FA_f":<12} {"Dstar":<12}')
    print('-' * 68)
    for s in SNR_list:
        mask = snr == s
        print(f'{int(s):<8} '
              f'{np.median(RE_MD_diff[mask]):<12.1f} '
              f'{np.median(RE_FA_diff[mask]):<12.1f} '
              f'{np.median(RE_Mf[mask]):<12.1f} '
              f'{np.median(RE_FA_f[mask]):<12.1f} '
              f'{np.median(RE_Dstar[mask]):<12.1f}')
elif model == 'model4':
    print(f'\n{"SNR":<8} {"MD_diff":<12} {"FA_diff":<12} {"Mf":<12} {"FA_f":<12} {"MD_pseudo":<12} {"FA_pseudo":<12}')
    print('-' * 80)
    for s in SNR_list:
        mask = snr == s
        print(f'{int(s):<8} '
              f'{np.median(RE_MD_diff[mask]):<12.1f} '
              f'{np.median(RE_FA_diff[mask]):<12.1f} '
              f'{np.median(RE_Mf[mask]):<12.1f} '
              f'{np.median(RE_FA_f[mask]):<12.1f} '
              f'{np.median(RE_MD_pseudo[mask]):<12.1f} '
              f'{np.median(RE_FA_pseudo[mask]):<12.1f}')
