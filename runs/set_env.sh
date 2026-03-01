# This script sets the NANOCHAT_CONFIG environment variable
# and prints the value of the environment variable
#
# Usage:
# source runs/set_env.sh
#
# This script is used to set the NANOCHAT_CONFIG environment variable
# and print the value of the environment variable
#


echo "Usage: source runs/set_env.sh"
echo "Warning: do not execute this script (i.e. bash runs/set_env.sh), but source it"

export NANOCHAT_CONFIG="configs/local.yaml"

echo "NANOCHAT_CONFIG: $NANOCHAT_CONFIG"
echo "Please run \"echo \$NANOCHAT_CONFIG\" in your terminal to verify the environment variable is set"
