#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LAUNCHER_CONFIG_PATH="${repo_root}/scripts/truncate/launcher_config.sh"
source "${LAUNCHER_CONFIG_PATH}"

stop_by_pid_file() {
  local name="$1"
  local pid_file="$2"

  if [[ ! -f "${pid_file}" ]]; then
    echo "${name}: no pid file at ${pid_file}"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    echo "${name}: empty pid file ${pid_file}"
    rm -f "${pid_file}"
    return 0
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    kill "${pid}"
    echo "${name}: sent SIGTERM to PID ${pid}"
  else
    echo "${name}: PID ${pid} not running"
  fi

  rm -f "${pid_file}"
}

stop_by_pid_file "truncate agent" "${repo_root}/pids/${AGENT_LAUNCH_NAME}.pid"
stop_by_pid_file "truncate mcp" "${repo_root}/pids/${MCP_LAUNCH_NAME}.pid"
stop_by_pid_file "truncate vllm" "${repo_root}/pids/${VLLM_LAUNCH_NAME}.pid"
stop_by_pid_file "truncate start_all" "${repo_root}/pids/${START_ALL_LAUNCH_NAME}.pid"
