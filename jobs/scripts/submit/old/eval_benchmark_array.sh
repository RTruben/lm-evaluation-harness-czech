#!/usr/bin/bash
#SBATCH --job-name bcm
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 12:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

# Define an array of tasks
export TASKS=(
  "benczechmark_propaganda_argumentace"
  "benczechmark_propaganda_fabulace"
  "benczechmark_propaganda_nazor"
  "benczechmark_propaganda_strach"
  "benczechmark_propaganda_zamereni"
  "benczechmark_propaganda_demonizace"
  "benczechmark_propaganda_lokace"
  "benczechmark_propaganda_relativizace"
  "benczechmark_propaganda_vina"
  "benczechmark_propaganda_zanr"
  "benczechmark_propaganda_emoce"
  "benczechmark_propaganda_nalepkovani"
  "benczechmark_propaganda_rusko"
  "benczechmark_sentiment_mall"
  "benczechmark_sentiment_fb"
  "benczechmark_sentiment_csfd"
  "benczechmark_summarization"
  "benczechmark_grammarerrorcorrection"
  "benczechmark_cs_naturalquestions"
  "benczechmark_cs_sqad32"
  "benczechmark_cs_triviaQA"
  "benczechmark_csfever_nli"
  "benczechmark_ctkfacts_nli"
  "benczechmark_cs_ner"
  "benczechmark_hellaswag"
  "benczechmark_histcorpus" # requires logprob summing, not averaging!
  "benczechmark_klokan_qa"
  "benczechmark_cs_court_decisions_ner"
  "benczechmark_umimeto_qa"
  "benczechmark_cermat_mc"
  "benczechmark_cermat_qa"
)

export HF_TOKEN="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"

SUM_LOGPROBS=(
  "benczechmark_histcorpus"
)

# Set up environment variables
export PYTHON=/scratch/project/open-28-72/ifajcik/mamba/envs/harness/bin/python
export TOKENIZERS_PARALLELISM=true
export HF_HOME="/home/ifajcik/data_scratch_new/hfhome"

export CACHE_NAME="realrun_benchmark_csmpt_cache_${TASKS[$SLURM_ARRAY_TASK_ID]}"
cd /home/ifajcik/data_scratch_new/lm-evaluation-harness || exit
export PYTHONPATH=$(pwd)

# Adjust the output path to include task-specific information
OUTPUT_PATH="results/eval_csmpt_test_$SLURM_ARRAY_TASK_ID"

# Set run script
# ./jobs/scripts/eval_csmpt_accelerate.sh if not in SUM_LOGPROBS else ./jobs/scripts/eval_csmpt_accelerate_sumlp.sh
SCRIPT="./jobs/scripts/models/eval_csmpt_accelerate.sh"
for task in "${SUM_LOGPROBS[@]}"; do
  if [ "$task" == "${TASKS[$SLURM_ARRAY_TASK_ID]}" ]; then
    SCRIPT="./jobs/scripts/models/eval_csmpt_accelerate_sumlp.sh"
    break
  fi
done

set -x # enables a mode of the shell where all executed commands are printed to the terminal
# Run the script with the task specified by SLURM_ARRAY_TASK_ID
time $SCRIPT "${TASKS[$SLURM_ARRAY_TASK_ID]}" "$OUTPUT_PATH" | tee -a "eval_csmpt_array_$SLURM_ARRAY_TASK_ID.log"
set +x
