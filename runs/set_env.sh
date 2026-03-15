# This script sets the shell environment for nanochat runs.
#
# Usage:
# source runs/set_env.sh
#
if [[ -n "${BASH_SOURCE[0]:-}" && "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Usage: source runs/set_env.sh"
  echo "Warning: do not execute this script with bash; source it so the exports persist"
  exit 1
fi

export OMP_NUM_THREADS=1
export NANOCHAT_CONFIG="configs/global.yaml"

echo "OMP_NUM_THREADS is set to be: $OMP_NUM_THREADS"
echo "NANOCHAT_CONFIG is set to be: $NANOCHAT_CONFIG"
echo "All runtime config is loaded from configs/global.yaml"
echo "Please run \"echo \$OMP_NUM_THREADS\" in your terminal to set the OMP_NUM_THREADS environment variable"
