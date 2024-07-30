#!/usr/bin/bash
#SBATCH --job-name bcm_mpt7b
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 16:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

NAME='mpt7b'
MODEL_NAME='mosaicml/mpt-7b'

source ./jobs/scripts/submit/fire/fire_S_smartt.sh
