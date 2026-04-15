#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher_config_path="${LAUNCHER_CONFIG_PATH:-${repo_root}/scripts/launcher_config.sh}"
source "${launcher_config_path}"

cmd=(
  "${PYTHON_BIN}"
  "${repo_root}/search_agent/qwen_client.py"
  --model "${AGENT_MODEL}"
  --model-server "${AGENT_MODEL_SERVER}"
  --mcp-url "${AGENT_MCP_URL}"
  --max_tokens "${AGENT_MAX_TOKENS}"
  --query-template "${QUERY_TEMPLATE}"
  --output-dir "${AGENT_OUTPUT_DIR}"
)

if [[ -n "${QUERY_FILE:-}" ]]; then
  cmd+=(--query "${QUERY_FILE}")
fi

bash "${repo_root}/scripts/launch_nohup.sh" \
  "${AGENT_LAUNCH_NAME}" \
  "${AGENT_LOG_PATH}" \
  "${cmd[@]}"
