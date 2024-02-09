MODEL_NAME='mistralai/Mixtral-8x7B-v0.1'
GPUs_per_model=8
echo "Executing in $(pwd)"

# run on .data/propaganda_zanr
# task from argument
TASK="$1"
OUTPUT_PATH="$2"
export TRANSFORMERS_CACHE="huggingface_cache"

set -x
lm_eval --model vllm \
  --model_args pretrained=$MODEL_NAME,tensor_parallel_size=$GPUs_per_model,dtype=bfloat16,gpu_memory_utilization=0.9,max_model_len=4096\
  --tasks "$TASK" \
  --batch_size 2 \
  --output_path "$OUTPUT_PATH" \
  --log_samples \
  --verbosity DEBUG
