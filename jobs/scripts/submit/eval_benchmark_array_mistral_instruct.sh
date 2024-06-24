#!/usr/bin/bash
#SBATCH --job-name bcm
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 16:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1


# Set up environment variables
export PYTHON=/scratch/project/open-28-72/ifajcik/mamba/envs/harness/bin/python
export TOKENIZERS_PARALLELISM=true
export HF_HOME="/home/ifajcik/data_scratch_new/hfhome"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Executing TASK: ${TASKS[$SLURM_ARRAY_TASK_ID]}"

export CACHE_NAME="realrun_benchmark_mistral_instruct_cache_${TASKS[$SLURM_ARRAY_TASK_ID]}"
cd /home/ifajcik/data_scratch_new/lm-evaluation-harness || exit
export PYTHONPATH=$(pwd)

source ./jobs/TASKS.sh
source ./jobs/HF_TOKEN.sh

# Adjust the output path to include task-specific information
OUTPUT_PATH="results/eval_mistral_instruct_${TASKS[$SLURM_ARRAY_TASK_ID]}"

# Set run script
SCRIPT="./jobs/scripts/models/eval_mistral_instruct_accelerate.sh"
for task in "${SUM_LOGPROBS[@]}"; do
  if [ "$task" == "${TASKS[$SLURM_ARRAY_TASK_ID]}" ]; then
    SCRIPT="./jobs/scripts/models/eval_mistral_instruct_accelerate_sumlp.sh"
    break
  fi
done

set -x # enables a mode of the shell where all executed commands are printed to the terminal
# Run the script with the task specified by SLURM_ARRAY_TASK_ID
time $SCRIPT "${TASKS[$SLURM_ARRAY_TASK_ID]}" "$OUTPUT_PATH" | tee -a "eval_mistral_instruct_array_${TASKS[$SLURM_ARRAY_TASK_ID]}.log"
set +x
