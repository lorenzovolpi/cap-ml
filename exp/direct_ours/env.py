import os

import quapy as qp

PROBLEM = "multiclass"
NUM_TEST = 1000
qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0
BASEDIR = os.path.join("output", "direct_ours")
JSON_PATH = os.path.join(BASEDIR, f"results_{PROBLEM}.json")
