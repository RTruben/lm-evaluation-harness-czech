#!/bin/bash

set -x
echo "Starting worker script at $(hostname)"
num_gpus=8
num_nodes="$((SLURM_JOB_NUM_NODES))"

# Starting ray
LOCAL_ADDR=$(hostname)
if [ "$MASTER_ADDR" != "$LOCAL_ADDR" ]; then
  echo "Connecting from $LOCAL_ADDR to Ray head at $MASTER_ADDR:$MASTER_PORT"
  ray start --address $MASTER_ADDR:$MASTER_PORT
fi


echo "Executing in $(pwd)"


export NUMEXPR_MAX_THREADS=$(nproc --all)

set -x

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

$PYTHON -m lm_eval --model vllm \
  --model_args pretrained=$MODEL_NAME,tensor_parallel_size=16,worker_use_ray=True,dtype=bfloat16,gpu_memory_utilization=0.8,max_length=2048,normalize_log_probs=$NORMALIZE_LOG_PROBS,trust_remote_code=True$TRUNCATE_STRATEGY_ARG \
  --tasks "$TASK" \
  --batch_size 1 \
  --output_path "$OUTPUT_PATH" \
  --log_samples \
  --verbosity DEBUG \
  --num_fewshot $NUM_FEWSHOT $CHAT_TEMPLATE_ARGS
