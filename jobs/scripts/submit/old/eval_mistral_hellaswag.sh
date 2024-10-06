#!/usr/bin/bash
#SBATCH --job-name eval
#SBATCH --account OPEN-28-55
#SBATCH --partition qgpu
#SBATCH --time 12:00:00
#SBATCH --gpus-per-node 8
#SBATCH --nodes 1

TASK="benczechmark_hellaswag"
OUTPUT_PATH="results/eval_mistral_hellaswag"
JOBSCRIPT="./jobs/scripts/eval_mistral_vllm.sh"

# export PYTHON
export PYTHON=/scratch/project/open-28-72/ifajcik/mamba/envs/harness/bin/python
export TOKENIZERS_PARALLELISM=true

# cd to the right directory
cd /home/ifajcik/data_scratch_new/lm-evaluation-harness || exit
chmod +rx $JOBSCRIPT  || exit


set -x # enables a mode of the shell where all executed commands are printed to the terminal
$JOBSCRIPT "$TASK" "$OUTPUT_PATH"  2>&1 | tee -a "eval_mistral7b_ourtrunc_hellaswag.log"
set +x
