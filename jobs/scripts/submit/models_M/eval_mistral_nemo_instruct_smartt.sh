#!/usr/bin/bash
#SBATCH --job-name bcm_mxtrl
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node 8
#SBATCH --nodes 1

NAME='mistral_nemo_instruct'
MODEL_NAME='mistralai/Mistral-Nemo-Base-2407'

source ./jobs/scripts/submit/fire/fire_M_smartt.sh
