#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Choose an explicit stop mode:"
echo "  bash ${repo_root}/scripts/truncate/stop_all.sh"
echo "  bash ${repo_root}/scripts/infmem/stop_all.sh"
exit 2
