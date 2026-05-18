#!/bin/bash
#SBATCH --job-name=IVIM-DTI-evaluate
#SBATCH --partition=luna-long
#SBATCH --mem=120G
#SBATCH --cpus-per-task=4
#SBATCH --time=7-00:00
#SBATCH --nice=10000
source ~/start.sh
cd /scratch/rnga/aoouladbrahim/IVIM-DTI-NET
echo 'Evaluating model3...'
python evaluate.py --model model3
