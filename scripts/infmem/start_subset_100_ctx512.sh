#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

export QUERY_FILE="${QUERY_FILE:-${repo_root}/topics-qrels/subsets/random100_seed42/queries.tsv}"
export MCP_SNIPPET_MAX_TOKENS="${MCP_SNIPPET_MAX_TOKENS:-512}"

effective_toolset="${TONGYI_TOOLSET:-}"
if [[ -z "${effective_toolset}" ]]; then
  if [[ "${MCP_INFMEM_POSITION:-search}" == "visit" ]]; then
    effective_toolset="custom"
  else
    effective_toolset="standard"
  fi
fi

toolset_suffix=""
if [[ "${effective_toolset}" == "custom" ]]; then
  toolset_suffix="_custom"
fi

infmem_position_suffix=""
case "${MCP_INFMEM_POSITION:-search}" in
  visit)
    infmem_position_suffix="_visit_infmem"
    ;;
  memory_control)
    infmem_position_suffix="_memctrl_infmem"
    ;;
esac

export RUN_TAG="${RUN_TAG:-${MCP_SEARCHER_TYPE:-bm25}_infmem${toolset_suffix}${infmem_position_suffix}_subset100_seed42_ctx512}"

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

bash "${repo_root}/scripts/infmem/start_all.sh"
