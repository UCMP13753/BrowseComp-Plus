#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export QUERY_FILE="${QUERY_FILE:-${repo_root}/topics-qrels/subsets/random100_seed42/queries.tsv}"
export MCP_SNIPPET_MAX_TOKENS="${MCP_SNIPPET_MAX_TOKENS:-512}"
toolset_suffix=""
if [[ "${TONGYI_TOOLSET:-standard}" == "custom" ]]; then
  toolset_suffix="_custom_visit"
fi
export RUN_TAG="${RUN_TAG:-${MCP_SEARCHER_TYPE:-bm25}_truncate${toolset_suffix}_subset100_seed42_ctx512}"

if [[ ! -f "${QUERY_FILE}" ]]; then
  subset_dir="$(dirname "${QUERY_FILE}")"
  python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
  if [[ ! -x "${python_bin}" ]]; then
    python_bin="python"
  fi

  "${python_bin}" "${repo_root}/scripts/prepare_query_subset.py" \
    --queries "${repo_root}/topics-qrels/queries.tsv" \
    --qrel-golds "${repo_root}/topics-qrels/qrel_golds.txt" \
    --qrel-evidence "${repo_root}/topics-qrels/qrel_evidence.txt" \
    --ground-truth "${repo_root}/data/browsecomp_plus_decrypted.jsonl" \
    --size 100 \
    --seed 42 \
    --output-dir "${subset_dir}"
fi

bash "${repo_root}/scripts/truncate/start_all.sh"
