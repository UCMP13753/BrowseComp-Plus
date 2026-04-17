#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pid_dir="${repo_root}/pids"
grace_seconds="${STOP_ALL_GRACE_SECONDS:-5}"

stop_pid_file() {
  local pid_file="$1"
  local name
  name="$(basename "${pid_file}" .pid)"

  if [[ ! -f "${pid_file}" ]]; then
    return 0
  fi

  local pid
  pid="$(tr -d '[:space:]' < "${pid_file}")"
  if [[ -z "${pid}" ]]; then
    echo "${name}: empty pid file, removing ${pid_file}"
    rm -f "${pid_file}"
    return 0
  fi

  if ! [[ "${pid}" =~ ^[0-9]+$ ]]; then
    echo "${name}: invalid pid '${pid}', removing ${pid_file}"
    rm -f "${pid_file}"
    return 0
  fi

  if kill -0 "${pid}" 2>/dev/null; then
    local cmd=""
    cmd="$(ps -p "${pid}" -o args= 2>/dev/null || true)"
    echo "${name}: sending SIGTERM to PID ${pid}${cmd:+ | ${cmd}}"
    kill "${pid}" 2>/dev/null || true
    remaining_pids+=("${pid}")
    remaining_names+=("${name}")
  else
    echo "${name}: PID ${pid} not running"
  fi

  rm -f "${pid_file}"
}

wait_and_force_kill() {
  local deadline=$((SECONDS + grace_seconds))
  local alive=1

  while (( SECONDS < deadline )); do
    alive=0
    for pid in "${remaining_pids[@]:-}"; do
      if kill -0 "${pid}" 2>/dev/null; then
        alive=1
        break
      fi
    done

    if (( alive == 0 )); then
      return 0
    fi

    sleep 1
  done

  local idx
  for idx in "${!remaining_pids[@]}"; do
    local pid="${remaining_pids[$idx]}"
    local name="${remaining_names[$idx]}"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "${name}: still alive after ${grace_seconds}s, sending SIGKILL to PID ${pid}"
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
}

shopt -s nullglob
pid_files=("${pid_dir}"/*.pid)
shopt -u nullglob

if [[ ! -d "${pid_dir}" || ${#pid_files[@]} -eq 0 ]]; then
  echo "No pid files found under ${pid_dir}"
  exit 0
fi

echo "Stopping all managed processes in ${pid_dir} ..."

remaining_pids=()
remaining_names=()

for pid_file in "${pid_files[@]}"; do
  stop_pid_file "${pid_file}"
done

if [[ ${#remaining_pids[@]} -gt 0 ]]; then
  wait_and_force_kill
fi

echo "All managed pid files have been processed."
