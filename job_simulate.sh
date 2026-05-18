#!/bin/bash
#SBATCH --job-name=IVIM-DTI-simulate
#SBATCH --partition=luna-long
#SBATCH --mem=50G
#SBATCH --cpus-per-task=12
#SBATCH --time=7-00:00
#SBATCH --nice=10000

# activate conda environment
source /home/rnga/aoouladbrahim/.conda/envs/ivimdti/bin/activate ivimdti

# go to project directory
cd /scratch/rnga/aoouladbrahim/IVIM-DTI-NET

# run simulation
echo 'Starting simulation...'
python simulate.py
echo 'Simulation done!'
