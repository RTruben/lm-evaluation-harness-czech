#!/usr/bin/bash
#SBATCH --job-name bcm_tllama
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 16:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

NAME='cstllama'
MODEL_NAME='BUT-FIT/CSTinyLlama-1.2B'

source ./jobs/scripts/submit/fire/fire_S_smartt.sh
