MODEL_NAME='BUT-FIT/CSMPT7b-100k' # set HF_TOKEN for private model!!!
num_gpus=$(nvidia-smi --query-gpu=count --format=csv,noheader | awk '{print $1}' | head -n 1)
GPUs_per_model=$num_gpus
echo "Executing in $(pwd)"

TASK="$1"
OUTPUT_PATH="$2"
export NUMEXPR_MAX_THREADS=$(nproc --all)

set -x
$PYTHON -m accelerate.commands.launch -m lm_eval --model hf \
  --model_args pretrained=$MODEL_NAME,dtype=bfloat16,max_length=2048,truncation=True,truncate_strategy=leave_description,trust_remote_code=True \
  --tasks "$TASK" \
  --batch_size $num_gpus \
  --output_path "$OUTPUT_PATH" \
  --log_samples \
  --use_cache $CACHE_NAME \
  --verbosity DEBUG
