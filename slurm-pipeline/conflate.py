import json
import os
import sys

PROJECT_SRC_PATH = os.path.realpath(os.path.join(__file__, "..", ".."))
sys.path.append(PROJECT_SRC_PATH)

from conflation.conflation import conflate  # noqa: E402

# function parameters are passed by slurm-pipeline via stdin
params = json.load(sys.stdin)
print(params)

conflate(**params)
