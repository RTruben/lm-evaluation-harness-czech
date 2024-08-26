echo "Executing in $(pwd)"

TASK="$1"
OUTPUT_PATH="$2"
SUMLOGP="$3"
CHAT_TEMPLATE="$4"
TRUNCATE_STRATEGY="$5"
NUM_FEWSHOT="$6"
MODEL_NAME="$7"

export NUMEXPR_MAX_THREADS=$(nproc --all)

set -x

# Check if OUTPUT_PATH exists
if [ -e "$OUTPUT_PATH" ]; then
  echo "Output path $OUTPUT_PATH already exists. Exiting."
  exit 0
fi


# Normalize log probs based on sumlogp argument
if [ "$SUMLOGP" = "no" ]; then
  NORMALIZE_LOG_PROBS="True"
else
  NORMALIZE_LOG_PROBS="False"
fi

# Chat template arguments based on chat_template argument
CHAT_TEMPLATE_ARGS=""
if [ "$CHAT_TEMPLATE" = "singleturn" ]; then
  CHAT_TEMPLATE_ARGS="--apply_chat_template"
elif [ "$CHAT_TEMPLATE" = "multiturn" ]; then
  CHAT_TEMPLATE_ARGS="--apply_chat_template --fewshot_as_multiturn"
fi

# Truncate strategy argument based on truncate_strategy argument
TRUNCATE_STRATEGY_ARG=""
if [ "$TRUNCATE_STRATEGY" != "none" ]; then
  TRUNCATE_STRATEGY_ARG=",truncate_strategy=$TRUNCATE_STRATEGY"
fi


$PYTHON \
  -m lm_eval --model hf \
  --model_args pretrained=$MODEL_NAME,dtype=bfloat16,parallelize=True,max_length=2048,truncation=True,normalize_log_probs=$NORMALIZE_LOG_PROBS,trust_remote_code=True$TRUNCATE_STRATEGY_ARG \
  --tasks "$TASK" \
  --batch_size 8\
  --output_path "$OUTPUT_PATH" \
  --log_samples \
  --verbosity DEBUG \
  --num_fewshot $NUM_FEWSHOT $CHAT_TEMPLATE_ARGS
