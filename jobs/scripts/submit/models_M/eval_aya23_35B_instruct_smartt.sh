#!/usr/bin/bash
#SBATCH --job-name bcm_aya35b
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node 4
#SBATCH --nodes 1

NAME='aya23_35b_instruct'
MODEL_NAME='CohereForAI/aya-23-35B'
BACKEND='huggingface'

source ./jobs/scripts/submit/fire/fire_M_smartt.sh
