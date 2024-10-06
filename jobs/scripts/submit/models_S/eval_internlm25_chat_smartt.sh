#!/usr/bin/bash
#SBATCH --job-name bcm_intlm
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 24:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

NAME='internlm2_5-7b-chat'
MODEL_NAME='internlm/internlm2_5-7b-chat'

source ./jobs/scripts/submit/fire/fire_S_smartt.sh

