#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash rsync-ckp-from-a3.sh

SRC_BASE="root@a3:/data/ldeng/code/nanochat-ascend/.cache/checkpoint"

echo "Start rsync checkpoints from A3: $SRC_BASE"

# Run from repository root so relative paths always resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure local destination exists.
mkdir -p ".cache/checkpoint"

# Pull checkpoints from A3 to local machine.
rsync -avz --info=progress2 --human-readable \
  "$SRC_BASE/base" \
  "$SRC_BASE/chatsft" \
  "$SRC_BASE/chatrl" \
  ".cache/checkpoint/"

echo "Successfully synced checkpoints from A3 to local .cache/checkpoint/"
