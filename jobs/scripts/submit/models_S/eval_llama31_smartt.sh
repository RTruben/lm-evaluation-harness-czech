#!/usr/bin/bash
#SBATCH --job-name bcm_llama31
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 16:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

NAME='llama31_lm'
MODEL_NAME='meta-llama/Meta-Llama-3.1-8B'

source ./jobs/scripts/submit/fire/fire_S_smartt.sh
