#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher_config_path="${LAUNCHER_CONFIG_PATH:-${repo_root}/scripts/launcher_config.sh}"
source "${launcher_config_path}"

searcher_type="${MCP_SEARCHER_TYPE}"
index_path="${MCP_INDEX_PATH}"
snippet_max_tokens="${MCP_SNIPPET_MAX_TOKENS}"
k="${MCP_K}"
long_doc_mode="${MCP_LONG_DOC_MODE}"
planning_port="${TONGYI_PORT}"
num_threads="${TONGYI_NUM_THREADS}"
temperature="${TONGYI_TEMPERATURE}"
top_p="${TONGYI_TOP_P}"
presence_penalty="${TONGYI_PRESENCE_PENALTY}"
model_name="${MCP_MODEL_NAME:-}"
normalize="${MCP_NORMALIZE:-0}"

cmd=(
  "${PYTHON_BIN}"
  "${repo_root}/search_agent/tongyi_client.py"
  --model "${AGENT_MODEL}"
  --output-dir "${AGENT_OUTPUT_DIR}"
  --searcher-type "${searcher_type}"
  --index-path "${index_path}"
  --port "${planning_port}"
  --long-doc-mode "${long_doc_mode}"
  --snippet-max-tokens "${snippet_max_tokens}"
  --k "${k}"
  --num-threads "${num_threads}"
  --temperature "${temperature}"
  --top_p "${top_p}"
  --presence_penalty "${presence_penalty}"
)

if [[ -n "${QUERY_FILE:-}" ]]; then
  cmd+=(--query "${QUERY_FILE}")
fi

if [[ "${searcher_type}" == "faiss" ]]; then
  if [[ -z "${model_name}" ]]; then
    echo "MCP_MODEL_NAME must be set when MCP_SEARCHER_TYPE=faiss" >&2
    exit 1
  fi
  cmd+=(--model-name "${model_name}")
  if [[ "${normalize}" == "1" ]]; then
    cmd+=(--normalize)
  fi
fi

if [[ "${TONGYI_STORE_RAW}" == "1" ]]; then
  cmd+=(--store-raw)
fi

if [[ "${long_doc_mode}" == "infmem" ]]; then
  cmd+=(
    --infmem-model "${MCP_INF_MEM_MODEL}"
    --infmem-model-server "${MCP_INF_MEM_MODEL_SERVER}"
    --infmem-timeout-seconds "${MCP_INF_MEM_TIMEOUT_SECONDS}"
    --infmem-max-recurrent-steps "${MCP_INF_MEM_MAX_RECURRENT_STEPS}"
  )
fi

bash "${repo_root}/scripts/launch_nohup.sh" \
  "${AGENT_LAUNCH_NAME}" \
  "${AGENT_LOG_PATH}" \
  "${cmd[@]}"
