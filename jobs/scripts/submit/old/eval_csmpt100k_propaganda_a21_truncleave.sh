#!/usr/bin/bash
#SBATCH --job-name eval
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 48:00:00
#SBATCH --gpus-per-node 8
#SBATCH --nodes 1


TASK="benczechmark_propaganda"
OUTPUT_PATH="results/eval_csmpt_propaganda_acc_truncleave"
JOBSCRIPT="./jobs/scripts/eval_csmpt_accelerate_truncleave.sh"


# export PYTHON
export PYTHON=/mnt/data/ifajcik/micromamba/envs/envs/lmharness/bin/python
export TOKENIZERS_PARALLELISM=true
export HF_HOME="/mnt/nvme/ifajcik/huggingface_cache"
export HF_TOKEN="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"
export CACHE_NAME="csmpt100k_propaganda_accelerate_a21_truncleave"
# cd to the right directory
cd /mnt/data/ifajcik/lm_harness || exit
chmod +rx $JOBSCRIPT  || exit


set -x # enables a mode of the shell where all executed commands are printed to the terminal
$JOBSCRIPT "$TASK" "$OUTPUT_PATH"  2>&1 | tee -a "eval_csmpt100k_propaganda_acc_truncleave.log"
set +x

