#!/usr/bin/bash
#SBATCH --job-name eval
#SBATCH --account OPEN-28-55
#SBATCH --partition qgpu
#SBATCH --time 12:00:00
#SBATCH --gpus-per-node 8
#SBATCH --nodes 1

#TASK="benczechmark_propaganda,benczechmark_sentiment,benczechmark_grammarerrorcorrection"
#TASK="benczechmark_propaganda,benczechmark_sentiment,benczechmark_grammarerrorcorrection"
TASK="benczechmark_histcorpus,benczechmark_cs_naturalquestions,benczechmark_cs_sqad32,benczechmark_cs_triviaQA,benczechmark_csfever_nli,benczechmark_ctkfacts_nli,benczechmark_cs_ner,benczechmark_hellaswag,benczechmark_histcorpus,benczechmark_klokan_qa,benczechmark_cs_court_decisions_ner,benczechmark_cs_sqad32,benczechmark_summarization,benczechmark_umimeto_qa"

OUTPUT_PATH="results/eval_cstllama_test"
JOBSCRIPT="./jobs/scripts/eval_cstllama_accelerate.sh"

# export PYTHON
export PYTHON=/mnt/minerva1/nlp-2/homes/ifajcik/micromamba/envs/harness/bin/python
export TOKENIZERS_PARALLELISM=true
export HF_HOME="/mnt/data/ifajcik/huggingface_cache"
export CACHE_NAME="cstllama_test_accelerate_pc9"

# cd to the right directory
cd /mnt/data/ifajcik/lm-evaluation-harness || exit
export PYTHONPATH=$(pwd)
chmod +rx $JOBSCRIPT || exit

set -x # enables a mode of the shell where all executed commands are printed to the terminal
$JOBSCRIPT "$TASK" "$OUTPUT_PATH"  | tee -a "eval_cstllama_test.log"
set +x
