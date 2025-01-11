#!/usr/bin/bash
#SBATCH --job-name bcm_qw2
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node 8
#SBATCH --nodes 1

NAME='70bqwen2_instruct'
MODEL_NAME='Qwen/Qwen2-72B-Instruct'
BACKEND='huggingface'



source ./jobs/scripts/submit/fire/fire_M_smartt.sh
