#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: bash scripts/launch_nohup.sh <name> <log_path> <command...>" >&2
  exit 1
fi

name="$1"
shift
log_path="$1"
shift

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
pid_dir="${repo_root}/pids"
mkdir -p "${pid_dir}" "$(dirname "${log_path}")"

pid_file="${pid_dir}/${name}.pid"

if [[ -f "${pid_file}" ]]; then
  old_pid="$(cat "${pid_file}")"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "${name} is already running with PID ${old_pid}"
    echo "Stop it with: kill ${old_pid}"
    exit 1
  fi
  rm -f "${pid_file}"
fi

nohup "$@" >"${log_path}" 2>&1 &
pid=$!
echo "${pid}" >"${pid_file}"

echo "Started ${name}"
echo "PID: ${pid}"
echo "PID file: ${pid_file}"
echo "Log: ${log_path}"
echo "Tail logs: tail -f ${log_path}"
echo "Stop: kill $(cat "${pid_file}")"
