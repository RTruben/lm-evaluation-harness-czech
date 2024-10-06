#!/usr/bin/bash
#SBATCH --job-name bcm_llama31
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node 8
#SBATCH --nodes 2

NAME='70bllama31_instruct'
MODEL_NAME='meta-llama/Meta-Llama-3.1-70B-Instruct'



source ./jobs/scripts/submit/fire/fire_L_smartt.sh
