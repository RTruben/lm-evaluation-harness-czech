#!/usr/bin/bash
#SBATCH --job-name bcm_phi3
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 16:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

NAME='phi3_instruct'
MODEL_NAME='microsoft/Phi-3-small-128k-instruct'

source ./jobs/scripts/submit/fire/fire_S_smartt.sh
