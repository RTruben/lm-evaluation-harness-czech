#!/usr/bin/bash
#SBATCH --job-name bcm_g2b
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 16:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

NAME='gemma2-2b'
MODEL_NAME='google/gemma-2-2b'

source ./jobs/scripts/submit/fire/fire_S_smartt.sh
