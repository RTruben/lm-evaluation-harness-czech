MODEL_NAME='mistralai/Mistral-7B-v0.1'
num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | awk '{print $1}' | head -n 1)
GPUs_per_model=$num_gpus
echo "Executing in $(pwd)"

# run on .data/propaganda_zanr
# task from argument
TASK="$1"
OUTPUT_PATH="$2"
export NUMEXPR_MAX_THREADS=$(nproc --all)

set -x
$PYTHON -m lm_eval --model vllm \
  --model_args pretrained=$MODEL_NAME,tensor_parallel_size=$GPUs_per_model,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096 --tasks "$TASK" \
  --batch_size 1 \
  --output_path "$OUTPUT_PATH" \
  --log_samples \
  --use_cache mistralcache_harness \
  --verbosity DEBUG
