#!/usr/bin/env bash

set -euo pipefail

export JAVA_HOME=/work/mingze/miniconda3/envs/browsecomp/lib/jvm
export JVM_PATH=/work/mingze/miniconda3/envs/browsecomp/lib/jvm/lib/server/libjvm.so
export PATH=$JAVA_HOME/bin:$PATH

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LAUNCHER_CONFIG_PATH="${repo_root}/scripts/infmem/launcher_config.sh"
source "${LAUNCHER_CONFIG_PATH}"

port_is_open() {
  local host="$1"
  local port="$2"
  (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1
}

wait_for_port() {
  local host="$1"
  local port="$2"
  local name="$3"
  local timeout="${4:-300}"
  local start_ts
  start_ts="$(date +%s)"

  echo "Waiting for ${name} on ${host}:${port} ..."
  while true; do
    if port_is_open "${host}" "${port}"; then
      echo "${name} is ready on ${host}:${port}"
      return 0
    fi

    if (( "$(date +%s)" - start_ts >= timeout )); then
      echo "Timed out waiting for ${name} on ${host}:${port}" >&2
      return 1
    fi
    sleep 2
  done
}

launch_answer_vllm() {
  if port_is_open "${ANSWER_VLLM_HOST}" "${ANSWER_VLLM_PORT}"; then
    echo "Answer vLLM already detected on ${ANSWER_VLLM_HOST}:${ANSWER_VLLM_PORT}, skipping launch."
    return 0
  fi

  echo "Launching answer vLLM for Tongyi DeepResearch ..."
  env \
    LAUNCHER_CONFIG_PATH="${LAUNCHER_CONFIG_PATH}" \
    VLLM_CUDA_VISIBLE_DEVICES="${ANSWER_VLLM_CUDA_VISIBLE_DEVICES}" \
    VLLM_HOST="${ANSWER_VLLM_HOST}" \
    VLLM_PORT="${ANSWER_VLLM_PORT}" \
    VLLM_MODEL_PATH="${ANSWER_MODEL_PATH}" \
    VLLM_TENSOR_PARALLEL_SIZE="${ANSWER_VLLM_TENSOR_PARALLEL_SIZE}" \
    VLLM_MAX_MODEL_LEN="${ANSWER_VLLM_MAX_MODEL_LEN}" \
    VLLM_REASONING_PARSER="${ANSWER_VLLM_REASONING_PARSER}" \
    VLLM_ROPE_SCALING="${ANSWER_VLLM_ROPE_SCALING}" \
    VLLM_ENABLE_REASONING="${ANSWER_VLLM_ENABLE_REASONING}" \
    VLLM_SERVED_MODEL_NAME="${ANSWER_VLLM_SERVED_MODEL_NAME}" \
    VLLM_GPU_MEMORY_UTILIZATION="${ANSWER_VLLM_GPU_MEMORY_UTILIZATION}" \
    VLLM_LOG_PATH="${ANSWER_VLLM_LOG_PATH}" \
    VLLM_LAUNCH_NAME="${ANSWER_VLLM_LAUNCH_NAME}" \
    bash "${repo_root}/scripts/vllm_serve.sh"
  wait_for_port "${ANSWER_VLLM_HOST}" "${ANSWER_VLLM_PORT}" "answer vLLM"
}

launch_infmem_vllm() {
  if port_is_open "${INFMEM_VLLM_HOST}" "${INFMEM_VLLM_PORT}"; then
    echo "InfMem vLLM already detected on ${INFMEM_VLLM_HOST}:${INFMEM_VLLM_PORT}, skipping launch."
    return 0
  fi

  echo "Launching InfMem vLLM for 4B memory model ..."
  env \
    LAUNCHER_CONFIG_PATH="${LAUNCHER_CONFIG_PATH}" \
    VLLM_CUDA_VISIBLE_DEVICES="${INFMEM_VLLM_CUDA_VISIBLE_DEVICES}" \
    VLLM_HOST="${INFMEM_VLLM_HOST}" \
    VLLM_PORT="${INFMEM_VLLM_PORT}" \
    VLLM_MODEL_PATH="${INFMEM_MODEL_PATH}" \
    VLLM_TENSOR_PARALLEL_SIZE="${INFMEM_VLLM_TENSOR_PARALLEL_SIZE}" \
    VLLM_MAX_MODEL_LEN="${INFMEM_VLLM_MAX_MODEL_LEN}" \
    VLLM_REASONING_PARSER="${INFMEM_VLLM_REASONING_PARSER}" \
    VLLM_ROPE_SCALING="${INFMEM_VLLM_ROPE_SCALING}" \
    VLLM_ENABLE_REASONING="${INFMEM_VLLM_ENABLE_REASONING}" \
    VLLM_SERVED_MODEL_NAME="${INFMEM_VLLM_SERVED_MODEL_NAME}" \
    VLLM_GPU_MEMORY_UTILIZATION="${INFMEM_VLLM_GPU_MEMORY_UTILIZATION}" \
    VLLM_LOG_PATH="${INFMEM_VLLM_LOG_PATH}" \
    VLLM_LAUNCH_NAME="${INFMEM_VLLM_LAUNCH_NAME}" \
    bash "${repo_root}/scripts/vllm_serve.sh"
  wait_for_port "${INFMEM_VLLM_HOST}" "${INFMEM_VLLM_PORT}" "InfMem vLLM"
}

launch_answer_vllm
launch_infmem_vllm

if port_is_open "${MCP_HOST}" "${MCP_PORT}"; then
  echo "MCP server already detected on ${MCP_HOST}:${MCP_PORT}, skipping launch."
else
  echo "Launching InfMem MCP server ..."
  bash "${repo_root}/scripts/mcp_serve.sh"
  wait_for_port "${MCP_HOST}" "${MCP_PORT}" "InfMem MCP server"
fi

echo "Launching InfMem agent run ..."
bash "${repo_root}/scripts/test_launcher.sh"

echo "InfMem launch steps completed."
echo "Answer model server: ${AGENT_MODEL_SERVER}"
echo "InfMem model server: ${MCP_INF_MEM_MODEL_SERVER}"
echo "Agent log: ${AGENT_LOG_PATH}"
echo "Tail agent log: tail -f ${AGENT_LOG_PATH}"
