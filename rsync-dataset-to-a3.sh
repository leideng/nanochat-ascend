#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash rsync-dataset-to-a3.sh


DEST_BASE="root@a3:/data/ldeng/code/nanochat-ascend/.cache/dataset"

echo "Start rsync dataset to A3: $DEST_BASE"

# Run from repository root so relative paths always resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Do not sync pretrain dataset; set that manually on A3.
rsync -avz --info=progress2 --human-readable \
  ".cache/dataset/eval" \
  ".cache/dataset/task" \
  "$DEST_BASE/"

echo "Successfully synced dataset in .cache to A3: $DEST_BASE"
