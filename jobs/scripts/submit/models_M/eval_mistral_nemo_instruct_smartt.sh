#!/usr/bin/bash
#SBATCH --job-name bcm_mnemo
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node 2
#SBATCH --nodes 1

NAME='mistral_nemo_instruct'
MODEL_NAME='mistralai/Mistral-Nemo-Instruct-2407'
BACKEND='huggingface'

source ./jobs/scripts/submit/fire/fire_M_smartt.sh
