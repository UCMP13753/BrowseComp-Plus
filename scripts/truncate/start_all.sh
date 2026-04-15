#!/usr/bin/env bash

set -euo pipefail

export JAVA_HOME=/work/mingze/miniconda3/envs/browsecomp/lib/jvm
export JVM_PATH=/work/mingze/miniconda3/envs/browsecomp/lib/jvm/lib/server/libjvm.so
export PATH=$JAVA_HOME/bin:$PATH

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LAUNCHER_CONFIG_PATH="${repo_root}/scripts/truncate/launcher_config.sh"
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

if port_is_open "${VLLM_HOST}" "${VLLM_PORT}"; then
  echo "vLLM server already detected on ${VLLM_HOST}:${VLLM_PORT}, skipping launch."
else
  echo "Launching truncate answer vLLM ..."
  bash "${repo_root}/scripts/vllm_serve.sh"
  wait_for_port "${VLLM_HOST}" "${VLLM_PORT}" "truncate answer vLLM"
fi

if port_is_open "${MCP_HOST}" "${MCP_PORT}"; then
  echo "MCP server already detected on ${MCP_HOST}:${MCP_PORT}, skipping launch."
else
  echo "Launching truncate MCP server ..."
  bash "${repo_root}/scripts/mcp_serve.sh"
  wait_for_port "${MCP_HOST}" "${MCP_PORT}" "truncate MCP server"
fi

echo "Launching truncate agent run ..."
bash "${repo_root}/scripts/test_launcher.sh"

echo "Truncate launch steps completed."
echo "Agent log: ${AGENT_LOG_PATH}"
echo "Tail agent log: tail -f ${AGENT_LOG_PATH}"
