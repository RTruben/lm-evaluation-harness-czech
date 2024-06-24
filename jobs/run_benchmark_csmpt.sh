#!/usr/bin/bash

source ./jobs/TASKS.sh

# sbatch array job, number of tasks is the length of the array TASKS
sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/eval_benchmark_array_csmpt.sh
