#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash rsync-a3.sh


DEST="root@a3:/data/ldeng/code/nanochat-ascend"

echo "Start to rsync dataset in .cache to A3 at $DEST"

# Run from repository root so relative paths always resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# do not rsycn pretrain dataset;
# manually set in a3 server
rsync -avz --info=progress2 --human-readable \
  ".cache/dataset/eval" \
  ".cache/dataset/task" \
  "$DEST"

echo "Successfully rsync dataset in .cache to A3 at $DEST"
