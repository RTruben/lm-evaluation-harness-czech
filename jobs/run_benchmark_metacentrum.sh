#!/usr/bin/bash

source ./jobs/TASKS.sh

# S MODELS - EACH JOB USES SINGLE GPU
# sbatch array job, number of tasks is the length of the array TASKS
sbatch -J 0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_cuni_mh.metacentrum.sh
