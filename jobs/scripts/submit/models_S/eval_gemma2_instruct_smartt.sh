#!/usr/bin/bash
#SBATCH --job-name bcm_gemma2
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 16:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

NAME='gemma2_instruct'
MODEL_NAME='google/gemma-2-9b-it'

source ./jobs/scripts/submit/fire/fire_S_smartt.sh

