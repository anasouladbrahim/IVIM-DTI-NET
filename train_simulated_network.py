
import os
import time
import numpy as np
import IVIMDTINET.deep as deep
import torch
from hyperparams import hyperparams as hp
import argparse

torch.manual_seed(0)
np.random.seed(0)

# nr of networks you want to train
repeats = 1

# parse model argument
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='model2', 
                    choices=['model1', 'model2', 'model3', 'model4'])
args_parsed = parser.parse_args()
model = args_parsed.model

# load De Jong's bval/bvec
bval = np.genfromtxt('data/dejong_bval.bval')
bvec = np.genfromtxt('data/dejong_bvec.bvec')

selsb = np.array(bval) == 0

# load simulated data
data = np.load(f'data/simulated_{model}_signals.npy')
S0 = np.nanmean(data[:, selsb], axis=1).astype('<f')
data = data / S0[:, None]


# remove NaN
res = [i for i, val in enumerate(data != data) if not val.any()]




# create folder to save trained networks
pathnn = 'trained_networks'
if not os.path.exists(pathnn):
    os.makedirs(pathnn)

arg = hp()
start_time = time.time()

if repeats > 1:
    for i in range(repeats):
        net = deep.learn_IVIM(data[res], bval, bvec, arg, model=model)
        torch.save(net.state_dict(), 
                  f'trained_networks/{arg.save_name}_simulated_{model}-{i}.pt')
else:
    net = deep.learn_IVIM(data[res], bval, bvec, arg, model=model)
    torch.save(net.state_dict(), 
              f'trained_networks/{arg.save_name}_simulated_{model}.pt')