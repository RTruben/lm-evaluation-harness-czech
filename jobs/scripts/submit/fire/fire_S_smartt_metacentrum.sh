# set up run settings
CHAT_TEMPLATE="none"
TRUNCATE_STRATEGY="leave_description"

# Set up environment variables
export PYTHON=/storage/brno12-cerit/home/hrabalm/workspace/lm-evaluation-harness/.venv/bin/python
export TOKENIZERS_PARALLELISM=true
# export HF_HOME="/home/ifajcik/data_scratch_new/hfhome"
export TMPDIR="$SCRATCHDIR"

# copy wandb and hugingface credentials
cp -p /storage/praha1/home/hrabalm/.netrc $HOME/.netrc
mkdir -p $HOME/.cache/huggingface
cp -p /storage/praha1/home/hrabalm/.cache/huggingface/token $HOME/.cache/huggingface/token

source ./jobs/TASKS.sh
# source ./jobs/HF_TOKEN.sh
source ./jobs/NUM_SHOT.sh

export CACHE_NAME="realrun_benczechmark_${NAME}_cache_${TASKS[$PBS_ARRAY_INDEX]}"
cd /storage/brno12-cerit/home/hrabalm/workspace/lm-evaluation-harness-czech || exit
export PYTHONPATH=$(pwd)

echo "PBS_ARRAY_INDEX: $PBS_ARRAY_INDEX"
echo "Executing TASK: ${TASKS[$PBS_ARRAY_INDEX]}"


# Set run script
SCRIPT="./jobs/scripts/models/eval_S_accelerate.sh"
SUM_LOGP_FLAG="no"
for task in "${SUM_LOGPROBS[@]}"; do
  if [ "$task" == "${TASKS[$PBS_ARRAY_INDEX]}" ]; then
    SUM_LOGP_FLAG="yes"
    CHAT_TEMPLATE="none"
    TRUNCATE_STRATEGY="none"
    NUM_FEWSHOT=0
    break
  fi
done

OUTPUT_PATH="results_hf/eval_${NAME}_${TASKS[$PBS_ARRAY_INDEX]}_chat_${CHAT_TEMPLATE}_trunc_${TRUNCATE_STRATEGY}"
LOGFILE="eval_${NAME}_array_${TASKS[$PBS_ARRAY_INDEX]}_chat_${CHAT_TEMPLATE}_trunc_${TRUNCATE_STRATEGY}.log"

set -x # enables a mode of the shell where all executed commands are printed to the terminal
# Run the script with the task specified by SLURM_ARRAY_TASK_ID
time $SCRIPT "${TASKS[$PBS_ARRAY_INDEX]}" "$OUTPUT_PATH" "$SUM_LOGP_FLAG" "$CHAT_TEMPLATE" "$TRUNCATE_STRATEGY" "$NUM_FEWSHOT" "$MODEL_NAME" | tee -a "$LOGFILE"
set +x
