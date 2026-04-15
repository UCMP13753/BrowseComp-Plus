#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export LAUNCHER_CONFIG_PATH="${repo_root}/scripts/infmem/launcher_config.sh"
source "${LAUNCHER_CONFIG_PATH}"

bash "${repo_root}/scripts/launch_nohup.sh" \
  "${START_ALL_LAUNCH_NAME}" \
  "${START_ALL_LOG_PATH}" \
  bash "${repo_root}/scripts/infmem/start_all.sh"
