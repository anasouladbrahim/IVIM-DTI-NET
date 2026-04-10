

import numpy as np

# Simulation settings
SNR_list  = list(range(5, 65, 5)) + [1000]
steps     = 3
MD_diff   = np.linspace(1.6, 2.2, steps + 1)    # [10^-3 mm^2/s]
Mf        = np.linspace(0.15, 0.25, steps + 1)   # [-]
MD_pseudo = np.linspace(20, 120, steps + 1)      # [10^-3 mm^2/s]
FAmax     = 0.5
N_sim     = 10

# load bval and bvec
bval = np.genfromtxt('data/dejong_bval.bval')
bvec = np.genfromtxt('data/dejong_bvec.bvec')


def getlambdas(MD, FAmax=0.5):
    
    # Sample random eigenvalues (assumed to be [1,0,0], [0,1,0], [0,0,1]) for a given MD with FA <= FAmax.
    
    while True:
        # Randomly sample 3 eigenvalues that sum to 3*MD (so mean = MD)
        lambda1 = np.random.uniform(0, 3 * MD) #
        lambda2 = np.random.uniform(0, 3 * MD - lambda1)
        lambda3 = 3 * MD - lambda1 - lambda2

        lambdas = np.array([lambda1, lambda2, lambda3])
        MD_calc = np.mean(lambdas)
        FA = np.sqrt(3/2) * np.sqrt(
            np.sum((lambdas - MD_calc)**2) / np.sum(lambdas**2)
        )

        if FA <= FAmax:
            break

    eigenvalues = np.sort(lambdas)[::-1]
    return eigenvalues, FA


def calc_bmat(bval, bvec):
    
    # Calculate b-matrix from b-values and gradient directions.
    
    gx = bvec[0, :]
    gy = bvec[1, :]
    gz = bvec[2, :]

    # bval * (g^T * g)
    bmat = bval[:, None] * np.column_stack([
        gx**2,                                  
        gy**2,
        gz**2,
        2 * gx * gy,
        2 * gx * gz,
        2 * gy * gz
    ])

    return bmat

def signal_model1(eigenvalues_D, bmat, S0=1.0):
    # Model 1: DTI only D tensor -> S = S0 * exp(-b * g^T * D * g)
    
    
    # D tensor elements (eigenvectors aligned with x,y,z axes)
    D_tensor = np.array([
        eigenvalues_D[0],  # Dxx = lambda1
        eigenvalues_D[1],  # Dyy = lambda2
        eigenvalues_D[2],  # Dzz = lambda3
        0.0,               # Dxy = 0
        0.0,               # Dxz = 0
        0.0                # Dyz = 0
    ]) * 1e-3              # convert to mm^2/s

    S = S0 * np.exp(-bmat @ D_tensor) # S0 * -(b * g^T * g) * D
    return S

def signal_model2(eigenvalues_D, eigenvalues_Dstar, f, bmat, S0=1.0):
    
    #Model 2: D tensor + D* tensor + f scalar -> S = S0 * (f * exp(-b * g^T * D* * g) + (1-f) * exp(-b * g^T * D * g))
    
    
    # D tensor elements
    D_tensor = np.array([
        eigenvalues_D[0],  # Dxx = lambda1
        eigenvalues_D[1],  # Dyy = lambda2
        eigenvalues_D[2],  # Dzz = lambda3
        0.0,               # Dxy = 0
        0.0,               # Dxz = 0
        0.0                # Dyz = 0
    ]) * 1e-3

    # D* tensor elements
    Dstar_tensor = np.array([
        eigenvalues_Dstar[0],  # Dxx* = lambda1
        eigenvalues_Dstar[1],  # Dyy* = lambda2
        eigenvalues_Dstar[2],  # Dzz* = lambda3
        0.0,                   # Dxy* = 0
        0.0,                   # Dxz* = 0
        0.0                    # Dyz* = 0
    ]) * 1e-3

    S = S0 * (f * np.exp(-bmat @ Dstar_tensor) + (1-f) * np.exp(-bmat @ D_tensor)) #S = S0 * (f * exp(-b * g^T * D* * g) + (1-f) * exp(-b * g^T * D * g))
    return S

def signal_model3(eigenvalues_D, eigenvalues_f, Dstar_scalar, bval, bmat, bvec, S0=1.0):
    #Model 3: D tensor + D* scalar + f tensor -> S = S0 * (g^T*f*g * exp(-b * D*) + (1 - g^T*f*g) * exp(-b * g^T * D * g))
    
    # D tensor elements
    D_tensor = np.array([
        eigenvalues_D[0],
        eigenvalues_D[1],
        eigenvalues_D[2],
        0.0, 0.0, 0.0
    ]) * 1e-3

    # f tensor elements
    f_tensor = np.array([
        eigenvalues_f[0], # fxx = lambda1 -> perfusion fraction in x direction
        eigenvalues_f[1], # fyy = lambda2 -> perfusion fraction in y direction
        eigenvalues_f[2], # fzz = lambda3 -> perfusion fraction in z direction
        0.0, 0.0, 0.0
    ])

    # direction matrix (bmat without b) for g^T * f * g
    gx = bvec[0, :]
    gy = bvec[1, :]
    gz = bvec[2, :]
    dir_matrix = np.column_stack([
        gx**2,
        gy**2,
        gz**2,
        2 * gx * gy,
        2 * gx * gz,
        2 * gy * gz
    ])  # shape (321, 6)

    # g^T * f * g for all 321 measurements
    gTfg = dir_matrix @ f_tensor  # shape (321,)

    # D* scalar
    Dstar = Dstar_scalar * 1e-3

    # S = S0 * (g^T*f*g * exp(-b * D*) + (1 - g^T*f*g) * exp(-b * g^T * D * g))
    S = S0 * (gTfg * np.exp(-bval * Dstar) + (1 - gTfg) * np.exp(-bmat @ D_tensor))  
    return S

def signal_model4(eigenvalues_D, eigenvalues_Dstar, eigenvalues_f, bmat, bvec, S0=1.0):
    #Model 4: D tensor + D* tensor + f tensor S = S0 * (g^T*f*g * exp(-b * g^T*D**g) + (1 - g^T*f*g) * exp(-b * g^T*D*g))
    
    # D tensor elements
    D_tensor = np.array([
        eigenvalues_D[0],
        eigenvalues_D[1],
        eigenvalues_D[2],
        0.0, 0.0, 0.0
    ]) * 1e-3

    # D* tensor elements
    Dstar_tensor = np.array([
        eigenvalues_Dstar[0],
        eigenvalues_Dstar[1],
        eigenvalues_Dstar[2],
        0.0, 0.0, 0.0
    ]) * 1e-3

    # f tensor elements
    f_tensor = np.array([
        eigenvalues_f[0],
        eigenvalues_f[1],
        eigenvalues_f[2],
        0.0, 0.0, 0.0
    ])

    # direction matrix for g^T * f * g
    gx = bvec[0, :]
    gy = bvec[1, :]
    gz = bvec[2, :]
    dir_matrix = np.column_stack([
        gx**2, gy**2, gz**2,
        2*gx*gy, 2*gx*gz, 2*gy*gz
    ])

    # g^T * f * g for all 321 measurements
    gTfg = dir_matrix @ f_tensor  # shape (321,)

    S = S0 * (gTfg * np.exp(-bmat @ Dstar_tensor) + (1 - gTfg) * np.exp(-bmat @ D_tensor))
    return S

bmat = calc_bmat(bval, bvec)
models = ['model1', 'model2', 'model3', 'model4']

def add_rician_noise(signal, SNR):
    
    sigma = 1.0 / SNR
    noise_real = np.random.normal(0, sigma, signal.shape)
    noise_imag = np.random.normal(0, sigma, signal.shape)
    noisy_signal = np.sqrt((signal + noise_real)**2 + noise_imag**2)
    return noisy_signal

for model in models:
    print(f'Simulating {model}...')
    
    # storage lists
    all_signals    = []  # noisy signals
    all_gt         = []  # ground truth parameters
    all_snr        = []  # SNR level per signal
    
    for md in MD_diff:
        for mf in Mf:
            for mds in MD_pseudo:
                
                # sample eigenvalues for each tensor
                eigenvalues_D,     FA_D     = getlambdas(md,  FAmax)
                eigenvalues_Dstar, FA_Dstar = getlambdas(mds, FAmax)
                eigenvalues_f,     FA_f     = getlambdas(mf,  FAmax)
                
                # compute true signal for this parameter combination
                if model == 'model1':
                    S_true = signal_model1(eigenvalues_D, bmat)
                elif model == 'model2':
                    S_true = signal_model2(eigenvalues_D, eigenvalues_Dstar, mf, bmat)
                elif model == 'model3':
                    S_true = signal_model3(eigenvalues_D, eigenvalues_f, mds, bval, bmat, bvec)
                elif model == 'model4':
                    S_true = signal_model4(eigenvalues_D, eigenvalues_Dstar, eigenvalues_f, bmat, bvec)
                
                # ground truth for this combination
                gt = {
                    'MD_diff':              md,
                    'FA_diff':              FA_D,
                    'eigenvalues_D':        eigenvalues_D,
                    'Mf':                   mf,
                    'FA_f':                 FA_f,
                    'eigenvalues_f':        eigenvalues_f,
                    'MD_pseudo':            mds,
                    'FA_pseudo':            FA_Dstar,
                    'eigenvalues_Dstar':    eigenvalues_Dstar
                }
                
                for snr in SNR_list:
                    for n in range(N_sim):
                        # add Rician noise
                        S_noisy = add_rician_noise(S_true, snr)
                        
                        # save
                        all_signals.append(S_noisy)
                        all_gt.append(gt)
                        all_snr.append(snr)
    
    # convert to numpy arrays
    all_signals = np.array(all_signals)  # shape (N, 321)
    all_snr     = np.array(all_snr)      # shape (N,)
    
    print(f'{model} done! Signals shape: {all_signals.shape}')
    
    # save
    np.save(f'data/simulated_{model}_signals.npy', all_signals)
    np.save(f'data/simulated_{model}_snr.npy', all_snr)
    np.save(f'data/simulated_{model}_gt.npy', all_gt)

print('Simulations done')