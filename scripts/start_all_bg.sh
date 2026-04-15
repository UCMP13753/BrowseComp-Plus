#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Choose an explicit background launch mode:"
echo "  bash ${repo_root}/scripts/truncate/start_all_bg.sh"
echo "  bash ${repo_root}/scripts/infmem/start_all_bg.sh"
exit 2
