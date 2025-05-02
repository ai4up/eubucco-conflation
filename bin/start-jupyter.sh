#!/bin/bash

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Scheduling Slurm job..."
jobid=$(sbatch --parsable ${SCRIPT_DIR}/submit-jupyter.sh)
workdir=$(squeue --format=%Z --noheader -j $jobid)
errfile=$(squeue --Format=STDERR:200 --noheader -j $jobid)
logfile=$(squeue --Format=STDOUT:200 --noheader -j $jobid)
errfile="${errfile//%j/$jobid}"
logfile="${logfile//%j/$jobid}"

echo -n "Waiting for Jupyter to start..."
sp="/-\|"
while [ ! -s $logfile ]
do
    job_state=$(squeue --states=all --noheader --format=%T -j $jobid)
    if [[ "$job_state" == "FAILED" || "$job_state" == "TIMEOUT" || "$job_state" == "CANCELLED" ]]; then
        echo "Job failed to start. Exiting."
        exit 1
    fi

    printf "\b${sp:i++%${#sp}:1}"
    sleep 1;
done

. $logfile

sleep 5
token=""
token=$(tail -1 $errfile | awk 'BEGIN { FS="="} ; { print $2 }')

echo ""
echo -e "\nJupyter notebook started successfully. To connect:"
echo "    1) Set up SHH tunnel to PIK cluster to forward Jupyter traffic:"
echo "    $FWDSSH"

if [ -z $token ]
then
  echo "    2) Access Jupyter notebook via: http://localhost:8888"
else
  echo "    2) Access Jupyter notebook via: http://localhost:8888?token=$token"
fi

echo -e "\nJupyter logs:"
tail -f $errfile