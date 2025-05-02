#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --qos=gpushort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100000
#SBATCH --time=24:00:00
#SBATCH --job-name=jupyter
#SBATCH --chdir=/p/tmp/floriann/jupyter-eubucco
#SBATCH --output=jupyter-%j.stdout
#SBATCH --error=jupyter-%j.stderr

set -eo pipefail

module purge
module load anaconda

for i in $(seq ${CONDA_SHLVL}); do
    source deactivate
done

source activate eubucco-features

# Select a random port for the notebook (needed if multiple notebooks are running on the same compute node)
NOTEBOOKPORT=`shuf -i 18000-18500 -n 1`

# Select a random port for tunneling (needed if multiple connections are happening on the same login node)
TUNNELPORT=`shuf -i 18501-19000 -n 1`

# Set up a reverse SSH tunnel from the compute node back to the login node (which is accessibile from the outside).
ssh -R $TUNNELPORT:localhost:$NOTEBOOKPORT login01 -N -f

# Instruction how to set up a forward SSH tunnel from the local computer to the login node.
echo "FWDSSH='ssh -J $(whoami)@hpc.pik-potsdam.de -L 8888:localhost:$TUNNELPORT $(whoami)@login01 -N'"

# Start the notebook
jupyter notebook --no-browser --port=$NOTEBOOKPORT --NotebookApp.token='' --NotebookApp.password=''

wait
