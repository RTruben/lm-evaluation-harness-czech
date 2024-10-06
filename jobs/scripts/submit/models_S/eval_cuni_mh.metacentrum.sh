#!/usr/bin/bash
#PBS -N llm_harness_cuni_mh
#PBS -q gpu
#PBS -l select=1:ncpus=1:mem=32gb:scratch_local=48gb:ngpus=1:gpu_cap=compute_70:gpu_mem=35gb
#PBS -l walltime=18:00:00
#PBS -m bae

NAME='CUNI-MH'
MODEL_NAME='/storage/brno12-cerit/home/hrabalm/workspace/wmt24_final_model'
SCRIPT_DIR="$PBS_O_WORKDIR"
cd "$SCRIPT_DIR" || exit

source ./jobs/scripts/submit/fire/fire_S_smartt_metacentrum.sh
