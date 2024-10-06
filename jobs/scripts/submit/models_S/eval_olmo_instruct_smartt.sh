#!/usr/bin/bash
#SBATCH --job-name bcm_olmins
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 16:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

NAME='olmo7b_instruct'
MODEL_NAME='allenai/OLMo-7B-Instruct-hf'

source ./jobs/scripts/submit/fire/fire_S_smartt.sh
