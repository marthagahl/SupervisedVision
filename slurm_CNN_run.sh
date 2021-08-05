#!/bin/bash

#SBATCH --time=8:00:00
#SBATCH --job-name=baseline
#SBATCH --output=/checkpoint/%u/jobs/baseline-%j.out
#SBATCH --error=/checkpoint/%u/jobs/baseline-%j.err

#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH -c 32
#SBATCH --mem=32G
#SBATCH --gpus-per-node=2
#SBATCH --ntasks=1

#SBATCH --ntasks-per-node=1


module purge
module unload anaconda3
module load anaconda3/4.3.1
module load cuda/10.1 cudnn/v7.6.5.32-cuda.10.1

source activate pytorch_env
 
python3 classifier_baseline.py --batch_size 24 --train_rotations 0 15 345 



