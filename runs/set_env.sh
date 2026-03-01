# This script sets the NANOCHAT_CONFIG environment variable
# and prints the value of the environment variable
#
# Usage:
# source runs/set_env.sh
#
# This script is used to set the global environment variable
#

if [[ -n "${BASH_SOURCE[0]:-}" && "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Usage: source runs/set_env.sh"
  echo "Warning: do not execute this script (i.e. bash runs/set_env.sh), but source it"
  exit 1
fi

export OMP_NUM_THREADS=1
export NANOCHAT_CONFIG="configs/local.yaml"

echo "OMP_NUM_THREADS is set to be: $OMP_NUM_THREADS"
echo "NANOCHAT_CONFI is set to be: $NANOCHAT_CONFIG"

echo "Please run \"echo \$NANOCHAT_CONFIG\" in your terminal to set the NANOCHAT_CONFIG environment variable"
echo "Please run \"echo \$OMP_NUM_THREADS\" in your terminal to set the OMP_NUM_THREADS environment variable"