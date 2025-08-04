#!/bin/bash

#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20000
#SBATCH --time=24:00:00
#SBATCH --job-name=candidate-pairs
#SBATCH --chdir=/p/projects/eubucco/data/conflation-training-data
#SBATCH --output=candidate-pairs-%j.stdout
#SBATCH --error=candidate-pairs-%j.stderr

pwd; hostname; date

module load anaconda

source activate eubucco-features

# python -u /p/projects/eubucco/eubucco-conflation/bin/create-candidate-pairs.py
# python -u /p/projects/eubucco/eubucco-conflation/bin/create-candidate-pairs-no-overlap.py
# python -u /p/projects/eubucco/eubucco-conflation/bin/create-candidate-neighborhoods.py
python -u /p/projects/eubucco/eubucco-conflation/bin/get-candidate-attributes.py
