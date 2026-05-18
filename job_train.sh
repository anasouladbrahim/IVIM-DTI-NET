#!/bin/bash
#SBATCH --job-name=IVIM-DTI-train
#SBATCH --partition=luna-long
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00
#SBATCH --nice=10000
source ~/start.sh
cd /scratch/rnga/aoouladbrahim/IVIM-DTI-NET
echo 'Starting training model3...'
python train_simulated_network.py --model model3
echo 'Training model3 done!'
