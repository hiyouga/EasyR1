#!/bin/bash
#SBATCH --job-name=easyr1
#SBATCH --output=slurm/%j_%x.out
#SBATCH --error=slurm/%j_%x.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=24:00:00


usage() {
  cat <<EOF
---------------------------------------------------------------------------
Configure via environment (defaults match the former per-model example scripts).

  MODEL_PATH       Hugging Face id or local path (required for real runs).

Optional:
  EXPERIMENT_NAME           Override run / checkpoint folder name.
  SAVE_CHECKPOINT_PATH      Override checkpoint folder path.

  Multi-node Ray (Slurm): set #SBATCH --nodes=N (N>1) and match GPUs per node. The script
  starts a Ray cluster (head + workers) like the Isambard distributed inference tutorial,
  then runs the VERL driver with RAY_ADDRESS set. Optional: RAY_PORT (default 6379), and on
  Isambard-AI export LOAD_BRICS_NCCL=1 to module load brics/nccl before ray start.

Submit examples:
  sbatch --export=ALL,MODEL_PATH=Qwen/Qwen3.5-9B examples/dist_math_grpo.sh
      
---------------------------------------------------------------------------

EOF
}
if [[ "$1" == "help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi


set -euo pipefail


MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
# MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-0.8B}"
SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}
SLURM_JOB_NUM_NODES=${SLURM_JOB_NUM_NODES:-1}


TRAINING_ROOT="${TRAINING_ROOT:-$(pwd)}"
cd "${TRAINING_ROOT}"

model_slug="$(basename "${MODEL_PATH}")"
model_slug="${model_slug//[^a-zA-Z0-9._-]/_}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-easyr1_${model_slug}_math_grpo}"

SAVE_ROOT="${SAVE_ROOT:-./checkpoints}"
SAVE_CHECKPOINT_PATH="${SAVE_CHECKPOINT_PATH:-${SAVE_ROOT}/${EXPERIMENT_NAME}}"
mkdir -p "${SAVE_CHECKPOINT_PATH}"

echo "Training root:  ${TRAINING_ROOT}"
echo "Model:          ${MODEL_PATH}"
echo "Experiment:     ${EXPERIMENT_NAME}"
echo "Checkpoints:    ${SAVE_CHECKPOINT_PATH}"

# Qwen3.5 on Hopper: Triton >= 3.4.0 refuses gated FLA backward (fla #640); TileLang is required.
# Pin apache-tvm-ffi==0.1.11 (tilelang aborts on import with 0.1.12).
if [[ "${MODEL_PATH}" == *[Qq]wen3.5* ]]; then
  export FLA_TILELANG=1
  echo "Qwen3.5: FLA_TILELANG=1 (Hopper + Triton 3.4+ requires TileLang for gated backward)"
fi

# vLLM torch.compile: shared ~/.cache/vllm on NFS can hit stale file handles (Errno 116)
# and FXGraphCacheMiss on multi-node Ray workers; use a per-job local cache.
VLLM_JOB_CACHE="${VLLM_JOB_CACHE:-/projects/u6gd/.cache/vllm_cache}"
mkdir -p "${VLLM_JOB_CACHE}"
export XDG_CACHE_HOME="${VLLM_JOB_CACHE}/xdg"
export TORCHINDUCTOR_CACHE_DIR="${VLLM_JOB_CACHE}/torchinductor"
mkdir -p "${XDG_CACHE_HOME}" "${TORCHINDUCTOR_CACHE_DIR}"
echo "vLLM/torch compile cache: ${VLLM_JOB_CACHE}"


export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"


########################################################
# Start ray cluster
# reduce the object store memory proportion to 5% 
export RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION=0.05

mapfile -t nodes_array < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
head_node="${nodes_array[0]}"
head_node_ip=$(srun --overlap --nodes=1 --ntasks=1 -w "$head_node" hostname -I | awk '{print $1}')
port=${RAY_PORT:-6379}




################## head node start ray cluster ##################

# Ray prestarts one Python worker per CPU; SLURM_CPUS_ON_NODE (e.g. 288) overwhelms raylet registration.
NUM_CPUS=${RAY_NUM_CPUS:-10}
echo "Starting HEAD at $head_node"
srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=6379 \
        --num-cpus "${NUM_CPUS}" --num-gpus "${SLURM_GPUS_PER_NODE}"  --dashboard-host=0.0.0.0 --block &
export RAY_ADDRESS="${head_node_ip}:${port}"
sleep 10

################## slave nodes start ray cluster ##################
# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    # the worker node blocks here.
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address="${RAY_ADDRESS}" --num-cpus "${NUM_CPUS}" --num-gpus "${SLURM_GPUS_PER_NODE}" --block &
    sleep 5
done


########################################################



echo CURRENT WORKING DIRECTORY: $(pwd)

# ray status
# ls -l /local/user
# echo $PATH

COMMON_ARGS=(
  "config=examples/config.yaml"
  "worker.actor.model.model_path=${MODEL_PATH}"
  "trainer.experiment_name=${EXPERIMENT_NAME}"
  "trainer.n_gpus_per_node=${SLURM_GPUS_PER_NODE}"
  "trainer.nnodes=${SLURM_JOB_NUM_NODES}"
  "trainer.save_checkpoint_path=${SAVE_CHECKPOINT_PATH}"
)

set -x

mkdir -p "${SAVE_CHECKPOINT_PATH}"

srun --overlap --nodes=1 --ntasks=1 -w "$head_node" \
  python -m verl.trainer.main "${COMMON_ARGS[@]}"  ${@} 2>&1 | tee "${SAVE_CHECKPOINT_PATH}/train.log"
