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

managed_pid_running() {
  local pid_file="$1"

  if [[ ! -f "${pid_file}" ]]; then
    return 1
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    return 1
  fi

  kill -0 "${pid}" 2>/dev/null
}

ensure_port_available_or_managed() {
  local host="$1"
  local port="$2"
  local name="$3"
  local pid_file="$4"

  if ! port_is_open "${host}" "${port}"; then
    return 0
  fi

  if managed_pid_running "${pid_file}"; then
    echo "${name} already detected on ${host}:${port}, skipping launch."
    return 10
  fi

  echo "${name} port ${host}:${port} is already occupied by an unmanaged process." >&2
  echo "Refusing to continue because vLLM may silently rebind to a different port." >&2
  echo "Please free the port first, or choose a different port in scripts/infmem/launcher_config.sh." >&2
  return 1
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

log_effective_endpoints() {
  echo "Configured Tongyi planning port: ${TONGYI_PORT}"
  echo "Configured answer server: ${ANSWER_MODEL_SERVER_URL}"
  echo "Configured InfMem server: ${MCP_INF_MEM_MODEL_SERVER}"
}

verify_vllm_bound_port() {
  local log_path="$1"
  local expected_port="$2"
  local name="$3"
  local bind_line=""
  local bound_port=""

  bind_line="$(grep -E "Starting vLLM API server on http://0\\.0\\.0\\.0:[0-9]+" "${log_path}" | tail -n 1 || true)"
  if [[ "${bind_line}" =~ http://0\.0\.0\.0:([0-9]+) ]]; then
    bound_port="${BASH_REMATCH[1]}"
  fi

  if [[ -z "${bound_port}" ]]; then
    echo "${name} did not log a final API server bind." >&2
    echo "Recent log excerpt:" >&2
    tail -n 20 "${log_path}" >&2
    return 1
  fi

  if [[ "${bound_port}" != "${expected_port}" ]]; then
    echo "${name} bound to port ${bound_port}, expected ${expected_port}." >&2
    echo "Recent log excerpt:" >&2
    tail -n 20 "${log_path}" >&2
    return 1
  fi
}

launch_answer_vllm() {
  local pid_file="${repo_root}/pids/${ANSWER_VLLM_LAUNCH_NAME}.pid"
  local ensure_status=0
  ensure_port_available_or_managed "${ANSWER_VLLM_HOST}" "${ANSWER_VLLM_PORT}" "Answer vLLM" "${pid_file}" || ensure_status=$?
  if [[ "${ensure_status}" -eq 10 ]]; then
    return 0
  elif [[ "${ensure_status}" -ne 0 ]]; then
    return "${ensure_status}"
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
}

launch_infmem_vllm() {
  local pid_file="${repo_root}/pids/${INFMEM_VLLM_LAUNCH_NAME}.pid"
  local ensure_status=0
  ensure_port_available_or_managed "${INFMEM_VLLM_HOST}" "${INFMEM_VLLM_PORT}" "InfMem vLLM" "${pid_file}" || ensure_status=$?
  if [[ "${ensure_status}" -eq 10 ]]; then
    return 0
  elif [[ "${ensure_status}" -ne 0 ]]; then
    return "${ensure_status}"
  fi

  echo "Launching InfMem vLLM for 4B memory model ..."
  env LAUNCHER_CONFIG_PATH="${LAUNCHER_CONFIG_PATH}" \
    bash "${repo_root}/scripts/infmem_vllm_serve.sh"
}

wait_for_vllm_stack() {
  wait_for_port "${ANSWER_VLLM_HOST}" "${ANSWER_VLLM_PORT}" "answer vLLM"
  verify_vllm_bound_port "${ANSWER_VLLM_LOG_PATH}" "${ANSWER_VLLM_PORT}" "Answer vLLM"

  wait_for_port "${INFMEM_VLLM_HOST}" "${INFMEM_VLLM_PORT}" "InfMem vLLM"
  verify_vllm_bound_port "${INFMEM_VLLM_LOG_PATH}" "${INFMEM_VLLM_PORT}" "InfMem vLLM"
}

log_effective_endpoints
launch_answer_vllm
launch_infmem_vllm
wait_for_vllm_stack

echo "Launching InfMem Tongyi run ..."
bash "${repo_root}/scripts/tongyi_launcher.sh"

echo "InfMem launch steps completed."
echo "Tongyi planning port: ${TONGYI_PORT}"
echo "Answer model server: ${ANSWER_MODEL_SERVER_URL}"
echo "InfMem model server: ${MCP_INF_MEM_MODEL_SERVER}"
echo "Agent log: ${AGENT_LOG_PATH}"
echo "Tail agent log: tail -f ${AGENT_LOG_PATH}"
