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
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_olmo_instruct_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_cstllama_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_gemma2-2b_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_gemma2-2b_instruct_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_gemma2_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_hermes_llama31_instruct_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_llama31_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_qwen2_smartt.sh

#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_llama3.2-1b_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_llama3.2-1b_instruct_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_llama3.2-3b_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_S/eval_llama3.2-3b_instruct_smartt.sh

# M MODELS - EACH JOB uses multiple GPUs
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_M/eval_mixtral8x7_instruct_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_M/eval_mistral_nemo_instruct_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_M/eval_mixtral8x7_base_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_M/eval_mistral_nemo_base_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_M/eval_aya23_35B_instruct_smartt.sh

# L MODELS - EACH JOB IS MULTINODE, and uses 8 GPUs/node
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_L/eval_llama31_instruct_70B_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_L/eval_llama31_70B_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_L/eval_mixtral_8x22B_instruct_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_L/eval_qwen2_instruct_70B_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_M/eval_qwen2_instruct_70B_smartt.sh # QWEN2 has OOM issues on some tasks for unknown reason, resubmit it as pipelined run with HF backend


# XXL MODELS - EACH JOB IS MANY MULTINODE (e.g. llama406 is 8 nodes), and uses 8 GPUs/node
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/models_XXL/eval_llama31_instruct_405B_smartt.sh
