MODEL_NAME='mistralai/Mistral-7B-Instruct-v0.2'
num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | awk '{print $1}' | head -n 1)
GPUs_per_model=$num_gpus
echo "Executing in $(pwd)"

# run on .data/propaganda_zanr
# task from argument
TASK="$1"
OUTPUT_PATH="$2"
export NUMEXPR_MAX_THREADS=$(nproc --all)

set -x
$PYTHON -m accelerate.commands.launch \
  --dynamo_backend=inductor \
  -m lm_eval --model hf \
  --model_args pretrained=$MODEL_NAME,dtype=bfloat16,max_length=2048,truncation=True,trust_remote_code=True \
  --tasks "$TASK" \
  --batch_size $num_gpus \
  --output_path "$OUTPUT_PATH" \
  --use_cache "$CACHE_NAME" \
  --log_samples \
  --verbosity DEBUG
