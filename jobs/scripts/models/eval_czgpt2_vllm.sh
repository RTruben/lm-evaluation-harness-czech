MODEL_NAME='BUT-FIT/Czech-GPT-2-XL-133k'
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
  --model_args pretrained=$MODEL_NAME,data_parallel_size=$GPUs_per_model,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=1024\
  --tasks "$TASK" \
  --output_path "$OUTPUT_PATH" \
  --log_samples \
  --use_cache gpt2cache_harness \
  --verbosity DEBUG
