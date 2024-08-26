#!/usr/bin/bash
#SBATCH --job-name bcm_hermes31
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 16:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

NAME='hermes_llama31_instruct'
MODEL_NAME='NousResearch/Hermes-3-Llama-3.1-8B'

source ./jobs/scripts/submit/fire/fire_S_smartt.sh


