
# IVIM-DTI-NET: Extended Tensor Models for Kidney Diffusion MRI

This repository extends [IVIM-DTI-NET](https://github.com/paulienvoorter/IVIM-DTI-NET)
(Voorter et al., 2025) to all four extended IVIM-DTI tensor models for kidney MRI
parameter estimation. It accompanies the MSc thesis *"Neural Networks for IVIM-DTI
Parameter Estimation in the Kidney: Robustness and Interpretability of Tensor Models
in Diffusion MRI"* (Anas Oulad Brahim, University of Amsterdam / Amsterdam UMC, 2026).

The physics-informed, self-supervised network is evaluated against Trust Region
Reflective least-squares fitting on simulated data with known ground truth, across a
range of signal-to-noise ratios. The work adds a gradient sensitivity (XAI) analysis
and a model-mismatch analysis to the original framework.

## The four tensor models

| Model | D | D\* | f | Free parameters |
|-------|--------|--------|--------|-----------------|
| 1 | tensor | — | — | 6 |
| 2 | tensor | tensor | scalar | 13 |
| 3 | tensor | scalar | tensor | 13 |
| 4 | tensor | tensor | tensor | 18 |

## Pipeline

```
simulate  →  train  →  evaluate  →  XAI / mismatch
```

Signals are simulated following the framework of de Jong (2022): 321 measurements
(one b = 0 plus 32 non-collinear directions at each of 10 non-zero b-values up to
600 s/mm²), 64 parameter combinations per model, Rician noise at 12 SNR levels from
5 to 60 plus SNR = 1000, with 10,000 noise realisations per combination.

## Attribution

This repository is a fork of Paulien Voorter's
[IVIM-DTI-NET](https://github.com/paulienvoorter/IVIM-DTI-NET), extended here to all
four tensor models and evaluated on simulated kidney data.

- Network architecture (IVIM-DTI-NET): Voorter et al. (2025), *Magn. Reson. Med.* 93:930–941.
- Simulation framework and acquisition protocol: de Jong (2022).
- Physics-informed self-supervised IVIM basis: Kaandorp et al. (2021).

The original code builds on earlier work, which the upstream authors gratefully acknowledge:

- Oliver Gurney-Champion and Misha Kaandorp — [IVIMNET](https://github.com/oliverchampion/IVIMNET)
- Sebastiano Barbieri — [deep_ivim](https://github.com/sebbarb/deep_ivim)
