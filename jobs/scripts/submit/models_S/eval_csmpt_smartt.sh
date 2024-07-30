#!/usr/bin/bash
#SBATCH --job-name bcm_csmpt7b
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 16:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

NAME='csmpt7b'
MODEL_NAME='BUT-FIT/csmpt7b'

source ./jobs/scripts/submit/fire/fire_S_smartt.sh
