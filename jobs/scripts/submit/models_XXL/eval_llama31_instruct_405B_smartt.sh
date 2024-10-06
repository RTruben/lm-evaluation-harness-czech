#!/usr/bin/bash
#SBATCH --job-name 405BLL
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node 8
#SBATCH --nodes 8

NAME='405bllama31_instruct'
MODEL_NAME='meta-llama/Meta-Llama-3.1-405B-Instruct'

source ./jobs/scripts/submit/fire/fire_L_smartt.sh
