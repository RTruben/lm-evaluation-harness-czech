#!/bin/bash

set -x
echo "Starting worker script at $(hostname)"
num_gpus=8
num_nodes="$((SLURM_JOB_NUM_NODES))"
WORKDIR="/home/ifajcik/data_scratch_new/lm-evaluation-harness"
cd $WORKDIR

# create temp directory for ray
hostname=$(hostname)
target_dir="${WORKDIR}/ray_${hostname}"
rm -rf /tmp/ray
rm -rf $target_dir
mkdir -p $target_dir
ln -s "$target_dir" /tmp/ray


source ~/.bashrc
micromamba activate harness

# Starting ray, else ray start --include-dashboard=True --head --port $MASTER_PORT
LOCAL_ADDR=$(hostname)
if [ "$MASTER_ADDR" != "$LOCAL_ADDR" ]; then
  sleep 10
  echo "Connecting from $LOCAL_ADDR to Ray head at $MASTER_ADDR:$MASTER_PORT"
  ray start --address $MASTER_ADDR:$MASTER_PORT
else
  echo "Starting Ray head at $MASTER_ADDR:$MASTER_PORT"
  ray start --include-dashboard=True --head --port $MASTER_PORT
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

#export VLLM_LOGGING_LEVEL=DEBUG
#export CUDA_LAUNCH_BLOCKING=1
#export NCCL_DEBUG=TRACE
#export VLLM_TRACE_FUNCTION=1

export NCCL_IB_GID_INDEX=3
#export TORCH_NCCL_USE_COMM_NONBLOCKING=1
#export NCCL_SOCKET_IFNAME=eth0

export NUMEXPR_MAX_THREADS=$(nproc --all)

TOTAL_GPUS=$((num_gpus * num_nodes))
# ONLY MASTER NODE SHOULD RUN THE EVALUATION, OTHERS SHOULD JUST WAIT
if [ "$MASTER_ADDR" == "$LOCAL_ADDR" ]; then
  sleep 15
   # Get the job array ID and index
  job_array_id=$SLURM_ARRAY_JOB_ID
  task_id=$SLURM_ARRAY_TASK_ID

  # Construct the full job ID
  full_job_id="${job_array_id}_${task_id}"

  # Print the job array task ID
  echo "Running job array task: $full_job_id"

  $PYTHON ./print_ray_nodes_once.py
  $PYTHON -m lm_eval --model vllm \
    --model_args pretrained=$MODEL_NAME,tensor_parallel_size=$TOTAL_GPUS,enforce_eager=True,worker_use_ray=True,dtype=bfloat16,gpu_memory_utilization=0.8,max_length=2048,normalize_log_probs=$NORMALIZE_LOG_PROBS,trust_remote_code=True$TRUNCATE_STRATEGY_ARG \
    --tasks "$TASK" \
    --batch_size 1 \
    --output_path "$OUTPUT_PATH" \
    --log_samples \
    --verbosity DEBUG \
    --num_fewshot $NUM_FEWSHOT $CHAT_TEMPLATE_ARGS

    # Cancel the current job array task
    echo "Completed job array task: $full_job_id. Running cancel command."
    scancel $full_job_id
else
  set +x
  echo "Waiting for master node to finish"
  while [ ! -f "$OUTPUT_PATH" ]; do
    sleep 10
  done
  echo "Master node finished, exiting"
fi
