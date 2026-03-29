#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash rsync-output-from-a3.sh

SRC_BASE="root@a3:/data/ldeng/code/nanochat-ascend/.cache/output"

echo "Start rsync output from A3: $SRC_BASE"

# Run from repository root so relative paths always resolve.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Ensure local destination exists.
mkdir -p ".cache/output"

# Pull the whole output directory from A3 to local machine.
rsync -avz --info=progress2 --human-readable \
  "$SRC_BASE/" \
  ".cache/output/"

echo "Successfully synced output from A3 to local .cache/output/"
