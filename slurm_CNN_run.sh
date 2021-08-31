#!/bin/bash

#SBATCH --time=1-12:00:00
#SBATCH --job-name=baseline
#SBATCH --output=/checkpoint/%u/jobs/baseline-%j.out
#SBATCH --error=/checkpoint/%u/jobs/baseline-%j.err

#SBATCH --partition=learnfair
#SBATCH --nodes=4
#SBATCH --mem=80G
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=10

#SBATCH --ntasks-per-node=4


module purge
module unload anaconda3
module load anaconda3/4.3.1
module load cuda/10.1 cudnn/v7.6.5.32-cuda.10.1

source activate pytorch_env
 
python3 classifier_baseline.py \
	--classes 16 \
	--workers 90 \
	--log_polar True \
	--lp_out_shape 190 165 \
	--salience False \
	--salience_points 4 \
	--training_aug 'random' \
	--dataset_path /checkpoint/mgahl/16_identities \
	--out_directory LogPolarParameters \
       	--experiment_name 190x165_3


