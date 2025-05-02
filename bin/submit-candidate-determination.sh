#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --qos=gpushort
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100000
#SBATCH --time=24:00:00
#SBATCH --job-name=candidate-pairs
#SBATCH --chdir=/p/projects/eubucco/data/conflation-training-data
#SBATCH --output=candidate-pairs-%j.stdout
#SBATCH --error=candidate-pairs-%j.stderr

pwd; hostname; date

module load anaconda

source activate eubucco-features

# python -u /p/projects/eubucco/eubucco-conflation/bin/create-candidate-pairs.py
python -u /p/projects/eubucco/eubucco-conflation/bin/create-candidate-neighborhoods.py
