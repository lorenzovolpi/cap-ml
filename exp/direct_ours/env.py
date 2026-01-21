import os

import quapy as qp

PROBLEM = "multiclass"
NUM_TEST = 1000
qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0
basedir = os.path.join("output", "direct_ours")
json_path = os.path.join(basedir, f"results_{PROBLEM}.json")
