#!/usr/bin/bash
#SBATCH --job-name benczechmark
#SBATCH --account OPEN-30-35
#SBATCH --partition qgpu
#SBATCH --time 3:00:00
#SBATCH --gpus-per-node 1
#SBATCH --nodes 1

# Define an array of tasks
TASKS=(
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
    "benczechmark_sentiment"
    "benczechmark_grammarerrorcorrection"
    "benczechmark_histcorpus"
    "benczechmark_cs_naturalquestions"
    "benczechmark_cs_sqad32"
    "benczechmark_cs_triviaQA"
    "benczechmark_csfever_nli"
    "benczechmark_ctkfacts_nli"
    "benczechmark_cs_ner"
    "benczechmark_hellaswag"
    "benczechmark_histcorpus"
    "benczechmark_klokan_qa"
    "benczechmark_cs_court_decisions_ner"
    "benczechmark_summarization"
    "benczechmark_umimeto_qa"
    "benczechmark_cermat"
)

# Set up environment variables
export PYTHON=/scratch/project/open-28-72/ifajcik/mamba/envs/harness/bin/python
export TOKENIZERS_PARALLELISM=true
export HF_HOME="/home/ifajcik/data_scratch_new/hfhome"
export HF_TOKEN="<YOUR_HF_TOKEN>"
export CACHE_NAME="benchmark"
cd /home/ifajcik/data_scratch_new/lm-evaluation-harness || exit
export PYTHONPATH=$(pwd)

# Adjust the output path to include task-specific information
OUTPUT_PATH="results/eval_csmpt_test_$SLURM_ARRAY_TASK_ID"

set -x # enables a mode of the shell where all executed commands are printed to the terminal
# Run the script with the task specified by SLURM_ARRAY_TASK_ID
time ./jobs/scripts/eval_csmpt_accelerate.sh "${TASKS[$SLURM_ARRAY_TASK_ID]}" "$OUTPUT_PATH" | tee -a "eval_csmpt_array_$SLURM_ARRAY_TASK_ID.log"
set +x
