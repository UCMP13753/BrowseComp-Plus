#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mode="${1:-${START_ALL_MODE:-truncate}}"

case "${mode}" in
  truncate|infmem)
    exec bash "${repo_root}/scripts/${mode}/start_all_bg.sh"
    ;;
  *)
    echo "Unknown background launch mode: ${mode}" >&2
    echo "Usage: bash ${repo_root}/scripts/start_all_bg.sh [truncate|infmem]" >&2
    exit 2
    ;;
esac
