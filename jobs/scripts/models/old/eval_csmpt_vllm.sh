MODEL_NAME='BUT-FIT/CSMPT7b-100k' # set HF_TOKEN for private model!!!
num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | awk '{print $1}' | head -n 1)
GPUs_per_model=$num_gpus
echo "Executing in $(pwd)"

TASK="$1"
OUTPUT_PATH="$2"
export NUMEXPR_MAX_THREADS=$(nproc --all)

set -x
$PYTHON -m lm_eval --model vllm \
  --model_args pretrained=$MODEL_NAME,tensor_parallel_size=$GPUs_per_model,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=2048 --tasks "$TASK" \
  --batch_size $num_gpus \
  --output_path "$OUTPUT_PATH" \
  --log_samples \
  --verbosity DEBUG
