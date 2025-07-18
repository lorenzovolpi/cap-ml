import os

import quapy as qp

import quacc as qc

PROJECT = "leap"
root_dir = os.path.join(qc.env["OUT_DIR"], PROJECT)
NUM_TEST = 1000
qp.environ["_R_SEED"] = 0
CSV_SEP = ","

_valid_problems = ["binary", "multiclass"]
PROBLEM = "binary"
