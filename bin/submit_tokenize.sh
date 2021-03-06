#! /bin/bash
#
# This script is intended to be used locally.
# It will create a new VM instance and submit the tokenize.sh script.

set -e

function usage() {
  echo "Usage: submit_tokenize.sh <subdir>"
  exit 1
}

if [ -z "$1" ]; then
  usage
fi

subdir="$1"
lowerSubdir=$(echo "$subdir" | tr '[:upper:]' '[:lower:]')
instanceName="cpu-sm-$lowerSubdir"
cmd="git clone https://github.com/ryandgoldenberg1/hotpot_ir.git && ./hotpot_ir/bin/tokenize.sh $subdir"

echo "Creating instance $instanceName"
gcloud compute instances create $instanceName \
  --source-instance-template cpu-sm

echo "Waiting for the instance to finish startup"
sleep 180s

echo "Submitting command to $instanceName: $cmd"
gcloud compute ssh rg3155@$instanceName \
  --ssh-key-file="~/.ssh/w4995_key" \
  --command="$cmd"

echo "Stopping instance $instanceName"
gcloud compute instances stop $instanceName
