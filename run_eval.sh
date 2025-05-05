#!/bin/bash

cd lm_eval && cd tasks && cd aver

DEVICE="cuda"
MODEL="google/mt5-base"
BATCH_SIZE=2
OUTPUT_PATH="aver_eval_results"

usage() {
  echo "Usage: $0 [-m <model>] [-o <output_path>] [-d <device>] [-b <batch_size>]"
  echo ""
  echo "  -m   Pretrained model (default: google/mt5-base)"
  echo "  -o   Output path (default: aver_eval_results)"
  echo "  -d   Device to use (default: cuda)"
  echo "  -b   Batch size (default: 2)"
  exit 1
}

while getopts ":m:o:d:b:" opt; do
  case $opt in
    m) MODEL="$OPTARG"
    ;;
    o) OUTPUT_PATH="$OPTARG"
    ;;
    d) DEVICE="$OPTARG"
    ;;
    b) BATCH_SIZE="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2; usage
    ;;
    :) echo "Option -$OPTARG requires an argument." >&2; usage
    ;;
  esac
done

python3 -m accelerate.commands.launch \
  --dynamo_backend=inductor \
  -m lm_eval \
  --model hf \
  --model_args pretrained="$MODEL",max_length=2048,truncation=True,normalize_log_probs=True,trust_remote_code=True,truncate_strategy=leave_description \
  --tasks aver_complete \
  --batch_size "$BATCH_SIZE" \
  --output_path "$OUTPUT_PATH" \
  --log_samples \
  --verbosity DEBUG \
  --device "$DEVICE"
