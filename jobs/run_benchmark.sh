#!/usr/bin/bash

source ./jobs/TASKS.sh

# S MODELS - EACH JOB USES SINGLE GPU
# sbatch array job, number of tasks is the length of the array TASKS
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_mpt7b_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_csmpt_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_llama3_instruct_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_llama31_instruct_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_gemma2_instruct_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_internlm25_chat_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_mistral_instruct03_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_aya23_instruct_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_phi3mini_instruct_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_qwen2_instruct_smartt.sh

# M MODELS - EACH JOB uses multiple GPUs
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_M/eval_mixtral8x7_instruct_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_M/eval_mistral_nemo_instruct_smartt.sh

# L MODELS - EACH JOB IS MULTINODE, and uses 8 GPUs/node
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_L/eval_llama31_instruct_TEST_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_L/eval_llama31_instruct_70B_smartt.sh