#!/usr/bin/bash

source ./jobs/TASKS.sh

# sbatch array job, number of tasks is the length of the array TASKS
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/eval_benchmark_array_llama3_instruct.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/eval_benchmark_array_llama3_instruct_smartt.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/eval_benchmark_array_llama3_instruct_smartt_chat.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/eval_benchmark_array_llama3_instruct_smartt_chat_multiturn.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/eval_benchmark_array_llama3_instruct_chat.sh
#sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/eval_benchmark_array_llama3_instruct_chat_multiturn.sh

sbatch --array=0-$((${#TASKS[@]} - 1)) jobs/scripts/submit/eval_benchmark_array_llama3_instruct_smartt_vllm.sh