#!/usr/bin/bash
#SBATCH --job-name bcm_mist03
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 16:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

NAME='mistral03_instruct'
MODEL_NAME='mistralai/Mistral-7B-Instruct-v0.3'

source ./jobs/scripts/submit/fire/fire_S_smartt.sh
