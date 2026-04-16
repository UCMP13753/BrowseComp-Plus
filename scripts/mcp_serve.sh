#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher_config_path="${LAUNCHER_CONFIG_PATH:-${repo_root}/scripts/launcher_config.sh}"
source "${launcher_config_path}"
export JAVA_HOME=/work/mingze/miniconda3/envs/browsecomp/lib/jvm
export JVM_PATH=/work/mingze/miniconda3/envs/browsecomp/lib/jvm/lib/server/libjvm.so
export PATH=$JAVA_HOME/bin:$PATH
cmd=(
  "${PYTHON_BIN}"
  "${repo_root}/searcher/mcp_server.py"
  --searcher-type "${MCP_SEARCHER_TYPE}"
  --index-path "${MCP_INDEX_PATH}"
  --port "${MCP_PORT}"
  --transport "${MCP_TRANSPORT}"
  --snippet-max-tokens "${MCP_SNIPPET_MAX_TOKENS}"
  --k "${MCP_K}"
)

if [[ "${MCP_SEARCHER_TYPE}" == "faiss" ]]; then
  if [[ -z "${MCP_MODEL_NAME:-}" ]]; then
    echo "MCP_MODEL_NAME must be set when MCP_SEARCHER_TYPE=faiss" >&2
    exit 1
  fi
  cmd+=(--model-name "${MCP_MODEL_NAME}")
  if [[ "${MCP_NORMALIZE:-0}" == "1" ]]; then
    cmd+=(--normalize)
  fi
fi

if [[ "${MCP_LONG_DOC_MODE}" == "infmem" ]]; then
  cmd+=(
    --long-doc-mode infmem
    --infmem-model "${MCP_INF_MEM_MODEL}"
    --infmem-model-server "${MCP_INF_MEM_MODEL_SERVER}"
    --infmem-timeout-seconds "${MCP_INF_MEM_TIMEOUT_SECONDS}"
    --infmem-max-recurrent-steps "${MCP_INF_MEM_MAX_RECURRENT_STEPS}"
  )
else
  cmd+=(--long-doc-mode "${MCP_LONG_DOC_MODE}")
fi

bash "${repo_root}/scripts/launch_nohup.sh" \
  "${MCP_LAUNCH_NAME}" \
  "${MCP_LOG_PATH}" \
  "${cmd[@]}"
