#!/usr/bin/bash
#SBATCH --job-name bcm_llama3
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 16:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

NAME='llama3_instruct'
MODEL_NAME='meta-llama/Meta-Llama-3-8B-Instruct'

source ./jobs/scripts/submit/fire/fire_S_smartt.sh
