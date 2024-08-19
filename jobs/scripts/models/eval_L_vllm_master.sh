echo "Starting master script at $(hostname)"

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()')

export TASK="$1"
export OUTPUT_PATH="$2"
export SUMLOGP="$3"
export CHAT_TEMPLATE="$4"
export TRUNCATE_STRATEGY="$5"
export NUM_FEWSHOT="$6"
export MODEL_NAME="$7"

source ~/.bashrc
micromamba activate harness

#ray start --head --port $MASTER_PORT


srun /home/ifajcik/data_scratch_new/lm-evaluation-harness/jobs/scripts/models/eval_L_vllm_worker.sh
echo "Finished master script at $(hostname)"