#!/usr/bin/bash
#SBATCH --job-name bcm_qwen2
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

NAME='qwen2_lm'
MODEL_NAME='Qwen/Qwen2-7B'

source ./jobs/scripts/submit/fire/fire_S_smartt.sh
