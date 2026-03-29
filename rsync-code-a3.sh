#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash rsync-a3.sh


DEST="root@a3:/data/ldeng/code/nanochat-ascend"

echo "Rsyncing code to A3 at $DEST"

# Run from repository root so relative paths always resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

rsync -avz \
  --exclude "__pycache__/" \
  --exclude "*.pyc" \
  "pyproject.toml" \
  ".python-version" \
  "configs" \
  "nanochat" \
  "runs" \
  "scripts" \
  "tasks" \
  "tests" \
  "$DEST"

echo "Done"