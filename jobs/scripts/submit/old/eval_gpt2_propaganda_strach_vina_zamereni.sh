#!/usr/bin/bash
#SBATCH --job-name eval
#SBATCH --account OPEN-28-55
#SBATCH --partition qgpu
#SBATCH --time 12:00:00
#SBATCH --gpus-per-node 8
#SBATCH --nodes 1

# strach, vina, zamereni

TASK="benczechmark_propaganda_strach"
OUTPUT_PATH="results/eval_gpt2_propaganda_strach"
JOBSCRIPT="./jobs/scripts/eval_czgpt2_vllm.sh"


# export PYTHON
export PYTHON=/scratch/project/open-28-72/ifajcik/mamba/envs/harness/bin/python
export TOKENIZERS_PARALLELISM=true

# cd to the right directory
cd /home/ifajcik/data_scratch_new/lm-evaluation-harness || exit
chmod +rx $JOBSCRIPT  || exit


set -x # enables a mode of the shell where all executed commands are printed to the terminal
$JOBSCRIPT "$TASK" "$OUTPUT_PATH"  2>&1 | tee -a "eval_gpt2_propaganda_strach.log"
set +x

TASK="benczechmark_propaganda_vina"
OUTPUT_PATH="results/eval_gpt2_propaganda_vina"
set -x
$JOBSCRIPT "$TASK" "$OUTPUT_PATH"  2>&1 | tee -a "eval_gpt2_propaganda_vina.log"
set +x

TASK="benczechmark_propaganda_zamereni"
OUTPUT_PATH="results/eval_gpt2_propaganda_zamereni"
set -x
$JOBSCRIPT "$TASK" "$OUTPUT_PATH"  2>&1 | tee -a "eval_gpt2_propaganda_zamereni.log"
set +x