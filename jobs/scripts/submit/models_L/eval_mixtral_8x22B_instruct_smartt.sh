#!/usr/bin/bash
#SBATCH --job-name bcm_8x22mix
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node 8
#SBATCH --nodes 2

NAME='8x22Mixtral_instruct'
MODEL_NAME='mistralai/Mixtral-8x22B-Instruct-v0.1'



source ./jobs/scripts/submit/fire/fire_L_smartt.sh
