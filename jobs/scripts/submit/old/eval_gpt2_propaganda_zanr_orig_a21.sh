#!/usr/bin/bash
#SBATCH --job-name eval
#SBATCH --account OPEN-28-55
#SBATCH --partition qgpu
#SBATCH --time 12:00:00
#SBATCH --gpus-per-node 8
#SBATCH --nodes 1
TASK="benczechmark_propaganda_zanr_orig"
OUTPUT_PATH="results/eval_gpt2_propaganda_zanrorig"
JOBSCRIPT="./jobs/scripts/eval_czgpt2_vllm.sh"


# export PYTHON
export PYTHON=/mnt/data/ifajcik/micromamba/envs/envs/lmharness/bin/python
export TOKENIZERS_PARALLELISM=true
export HF_HOME="/mnt/nvme/ifajcik/huggingface_cache"


# cd to the right directory
cd /mnt/data/ifajcik/lm_harness || exit
chmod +rx $JOBSCRIPT  || exit

set -x # enables a mode of the shell where all executed commands are printed to the terminal
$JOBSCRIPT "$TASK" "$OUTPUT_PATH"
set +x

