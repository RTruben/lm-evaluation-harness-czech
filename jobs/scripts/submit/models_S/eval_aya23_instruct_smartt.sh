#!/usr/bin/bash
#SBATCH --job-name bcm_aya
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 16:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

NAME='aya23_instruct'
MODEL_NAME='CohereForAI/aya-23-8B'

source ./jobs/scripts/submit/fire/fire_S_smartt.sh
