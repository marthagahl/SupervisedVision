#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --job-name=salience
#SBATCH --output=/checkpoint/%u/jobs/salience-%j.out
#SBATCH --error=/checkpoint/%u/jobs/salience-%j.err

#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --cpus-per-task 8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=8
#SBATCH --ntasks=8

#SBATCH --ntasks-per-node=8


module purge
module unload anaconda3
module load anaconda3/4.3.1
module load cuda/10.1 cudnn/v7.6.5.32-cuda.10.1

source activate tensorflow_env
 
python3 make_salience_maps_val.py



