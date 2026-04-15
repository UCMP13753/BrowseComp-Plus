#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher_config_path="${LAUNCHER_CONFIG_PATH:-${repo_root}/scripts/launcher_config.sh}"
source "${launcher_config_path}"

cmd=(
  env "CUDA_VISIBLE_DEVICES=${INFMEM_VLLM_CUDA_VISIBLE_DEVICES}"
  vllm serve "${INFMEM_MODEL_PATH}"
  --port "${INFMEM_VLLM_PORT}"
  --max-model-len "${INFMEM_VLLM_MAX_MODEL_LEN}"
  --tensor-parallel-size "${INFMEM_VLLM_TENSOR_PARALLEL_SIZE}"
)

if [[ "${INFMEM_VLLM_ENABLE_REASONING}" == "1" ]]; then
  cmd+=(--enable-reasoning --reasoning-parser "${INFMEM_VLLM_REASONING_PARSER}")
fi

if [[ -n "${INFMEM_VLLM_ROPE_SCALING}" ]]; then
  cmd+=(--rope-scaling "${INFMEM_VLLM_ROPE_SCALING}")
fi

if [[ -n "${INFMEM_VLLM_SERVED_MODEL_NAME}" ]]; then
  cmd+=(--served-model-name "${INFMEM_VLLM_SERVED_MODEL_NAME}")
fi

if [[ -n "${INFMEM_VLLM_GPU_MEMORY_UTILIZATION}" ]]; then
  cmd+=(--gpu-memory-utilization "${INFMEM_VLLM_GPU_MEMORY_UTILIZATION}")
fi

bash "${repo_root}/scripts/launch_nohup.sh" \
  "${INFMEM_VLLM_LAUNCH_NAME}" \
  "${INFMEM_VLLM_LOG_PATH}" \
  "${cmd[@]}"
