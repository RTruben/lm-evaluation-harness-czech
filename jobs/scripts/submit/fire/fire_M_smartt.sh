# set up run settings
CHAT_TEMPLATE="none"
TRUNCATE_STRATEGY="leave_description"

# Set up environment variables
export PYTHON=/scratch/project/open-30-35/ifajcik/mamba/envs/harness/bin/python
export TOKENIZERS_PARALLELISM=true
export HF_HOME="/home/ifajcik/data_scratch_new/hfhome"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

source ./jobs/TASKS.sh
source ./jobs/HF_TOKEN.sh
source ./jobs/NUM_SHOT.sh

export CACHE_NAME="realrun_benczechmark_${NAME}_cache_${TASKS[$SLURM_ARRAY_TASK_ID]}"
cd /home/ifajcik/data_scratch_new/lm-evaluation-harness || exit
export PYTHONPATH=$(pwd)

echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "Executing TASK: ${TASKS[$SLURM_ARRAY_TASK_ID]}"

ml CUDA/12.1.1

# Set run script
if [ "$BACKEND" == "vllm" ]; then
  SCRIPT="./jobs/scripts/models/eval_M_vllm.sh"
else
  SCRIPT="./jobs/scripts/models/eval_M_accelerate.sh"
fi
SUM_LOGP_FLAG="no"
for task in "${SUM_LOGPROBS[@]}"; do
  if [ "$task" == "${TASKS[$SLURM_ARRAY_TASK_ID]}" ]; then
    SUM_LOGP_FLAG="yes"
    CHAT_TEMPLATE="none"
    TRUNCATE_STRATEGY="none"
    NUM_FEWSHOT=0
    break
  fi
done

OUTPUT_PATH="results_hf/eval_${NAME}_${TASKS[$SLURM_ARRAY_TASK_ID]}_chat_${CHAT_TEMPLATE}_trunc_${TRUNCATE_STRATEGY}"
LOGFILE="eval_${NAME}_array_${TASKS[$SLURM_ARRAY_TASK_ID]}_chat_${CHAT_TEMPLATE}_trunc_${TRUNCATE_STRATEGY}.log"

set -x # enables a mode of the shell where all executed commands are printed to the terminal
# Run the script with the task specified by SLURM_ARRAY_TASK_ID
time $SCRIPT "${TASKS[$SLURM_ARRAY_TASK_ID]}" "$OUTPUT_PATH" "$SUM_LOGP_FLAG" "$CHAT_TEMPLATE" "$TRUNCATE_STRATEGY" "$NUM_FEWSHOT" "$MODEL_NAME" | tee -a "$LOGFILE"
set +x
