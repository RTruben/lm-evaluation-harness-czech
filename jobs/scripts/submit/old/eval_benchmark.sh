#!/usr/bin/bash
#SBATCH --job-name eval
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 24:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

#TASK="benczechmark_propaganda,benczechmark_sentiment,benczechmark_grammarerrorcorrection,benczechmark_histcorpus,benczechmark_cs_naturalquestions,benczechmark_cs_sqad32,benczechmark_cs_triviaQA,benczechmark_csfever_nli,benczechmark_ctkfacts_nli,benczechmark_cs_ner,benczechmark_hellaswag,benczechmark_histcorpus,benczechmark_klokan_qa,benczechmark_cs_court_decisions_ner,benczechmark_cs_sqad32,benczechmark_summarization,benczechmark_umimeto_qa"
TASK="benczechmark_cs_court_decisions_ner"
OUTPUT_PATH="results/eval_csmpt_test"
JOBSCRIPT="./jobs/scripts/eval_csmpt_accelerate.sh"

# export PYTHON
export PYTHON=/scratch/project/open-28-72/ifajcik/mamba/envs/harness/bin/python


export TOKENIZERS_PARALLELISM=true
export HF_HOME="/home/ifajcik/data_scratch_new/hfhome"
export CACHE_NAME="benchmark"

# cd to the right directory
cd /home/ifajcik/data_scratch_new/lm-evaluation-harness || exit
export PYTHONPATH=$(pwd)
chmod +rx $JOBSCRIPT || exit


set -x # enables a mode of the shell where all executed commands are printed to the terminal
time $JOBSCRIPT "$TASK" "$OUTPUT_PATH"  | tee -a "eval_csmpt_test.log"
set +x
