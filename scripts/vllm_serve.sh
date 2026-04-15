#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher_config_path="${LAUNCHER_CONFIG_PATH:-${repo_root}/scripts/launcher_config.sh}"
source "${launcher_config_path}"

VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}"
VLLM_ROPE_SCALING="${VLLM_ROPE_SCALING:-}"

cmd=(
  env "CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES}"
  vllm serve "${VLLM_MODEL_PATH}"
  --port "${VLLM_PORT}"
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE}"
)

if [[ "${VLLM_ENABLE_REASONING}" == "1" ]]; then
  cmd+=(--enable-reasoning --reasoning-parser "${VLLM_REASONING_PARSER}")
fi

if [[ -n "${VLLM_ROPE_SCALING}" ]]; then
  cmd+=(--rope-scaling "${VLLM_ROPE_SCALING}")
fi

if [[ -n "${VLLM_SERVED_MODEL_NAME}" ]]; then
  cmd+=(--served-model-name "${VLLM_SERVED_MODEL_NAME}")
fi

if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION}" ]]; then
  cmd+=(--gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}")
fi

bash "${repo_root}/scripts/launch_nohup.sh" \
  "${VLLM_LAUNCH_NAME}" \
  "${VLLM_LOG_PATH}" \
  "${cmd[@]}"

# nohup env CUDA_VISIBLE_DEVICES=0,1,2,3  vllm serve YuWangX/Memalpha-4B \
#     --tensor_parallel_size 4 \
#     --gpu-memory-utilization 0.9 \
#     --served-model-name qwen \
#     --port 8000 \
#     > logs/vllm_4b.log 2>&1 &

# nohup env CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve /work/mingze/models/Tongyi-DeepResearch-30B-A3B \
#     --tensor_parallel_size 4 \
#     --gpu-memory-utilization 0.9 \
#     --served-model-name qwen32b \
#     --port 8001 \
#     > logs/vllm_qwen32b.log 2>&1 &

# /work/mingze/models/qwen4b_infmem_earlystop_stage2/global_step_150/huggingface 
