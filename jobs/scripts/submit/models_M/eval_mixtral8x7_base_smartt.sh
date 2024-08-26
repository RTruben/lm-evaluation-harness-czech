#!/usr/bin/bash
#SBATCH --job-name bcm_mmix
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node 8
#SBATCH --nodes 1

NAME='mixtralM_base'
MODEL_NAME='mistralai/Mixtral-8x7B-v0.1'
BACKEND='vllm'

source ./jobs/scripts/submit/fire/fire_M_smartt.sh
