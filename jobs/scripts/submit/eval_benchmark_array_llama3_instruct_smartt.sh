#!/usr/bin/bash
#SBATCH --job-name bcm
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 16:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

# set up run settings
CHAT_TEMPLATE="none"
TRUNCATE_STRATEGY="leave_description"

# Set up environment variables
export PYTHON=/scratch/project/open-30-35/ifajcik/mamba/envs/harness/bin/python
export TOKENIZERS_PARALLELISM=true
export HF_HOME="/home/ifajcik/data_scratch_new/hfhome"

source ./jobs/TASKS.sh
source ./jobs/HF_TOKEN.sh

export CACHE_NAME="realrun_benchmark_llama3_instruct_cache_${TASKS[$SLURM_ARRAY_TASK_ID]}"
cd /home/ifajcik/data_scratch_new/lm-evaluation-harness || exit
export PYTHONPATH=$(pwd)

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Executing TASK: ${TASKS[$SLURM_ARRAY_TASK_ID]}"

# Set run script
SCRIPT="./jobs/scripts/models/eval_llama3_instruct_accelerate.sh"
SUM_LOGP_FLAG="no"
for task in "${SUM_LOGPROBS[@]}"; do
  if [ "$task" == "${TASKS[$SLURM_ARRAY_TASK_ID]}" ]; then
    SUM_LOGP_FLAG="yes"
    TRUNCATE_STRATEGY="none"
    break
  fi
done

OUTPUT_PATH="results/eval_llama3_instruct_${TASKS[$SLURM_ARRAY_TASK_ID]}_chat_${CHAT_TEMPLATE}_trunc_${TRUNCATE_STRATEGY}"
LOGFILE="eval_llama3_instruct_array_${TASKS[$SLURM_ARRAY_TASK_ID]}_chat_${CHAT_TEMPLATE}_trunc_${TRUNCATE_STRATEGY}.log"

set -x # enables a mode of the shell where all executed commands are printed to the terminal
# Run the script with the task specified by SLURM_ARRAY_TASK_ID
time $SCRIPT "${TASKS[$SLURM_ARRAY_TASK_ID]}" "$OUTPUT_PATH" "$SUM_LOGP_FLAG" "$CHAT_TEMPLATE" "$TRUNCATE_STRATEGY" | tee -a "$LOGFILE"
set +x
