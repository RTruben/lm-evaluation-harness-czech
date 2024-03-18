MODEL_NAME='BUT-FIT/CSTinyLlama-1.2B'
num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | awk '{print $1}' | head -n 1)
GPUs_per_model=$num_gpus
batch_size=8
echo "Executing in $(pwd)"

TASK="$1"
OUTPUT_PATH="$2"
export NUMEXPR_MAX_THREADS=$(nproc --all)

set -x
$PYTHON -m accelerate.commands.launch \
  --dynamo_backend=inductor \
-m lm_eval --model hf \
  --model_args pretrained=$MODEL_NAME,dtype=bfloat16,max_length=2048,truncation=True,trust_remote_code=True \
  --tasks "$TASK" \
  --batch_size $(($num_gpus * $batch_size)) \
  --output_path "$OUTPUT_PATH" \
  --log_samples \
  --use_cache $CACHE_NAME \
  --verbosity DEBUG
