import os

import quapy as qp

import cap

cap.env["OUT_DIR"] = os.getenv("EXP_OUT_DIR", "./output")
cap.env["QUACC_DATA"] = os.getenv("QUACC_DATA", cap.env["QUACC_DATA"])
cap.env["QUAPY_DATA"] = os.getenv("QUACC_QUAPY_DATA", cap.env["QUAPY_DATA"])
cap.env["SKLEARN_DATA"] = os.getenv("QUACC_SKLEARN_DATA", cap.env["SKLEARN_DATA"])
cap.env["N_JOBS"] = int(os.getenv("QUACC_N_JOBS", cap.env["N_JOBS"]))
_force_njobs = os.getenv("QUACC_FORCE_NJOBS", None)
if _force_njobs is not None:
    cap.env["FORCE_NJOBS"] = int(_force_njobs) > 0


PROJECT = "leap"
root_dir = os.path.join(cap.env["OUT_DIR"], PROJECT)
NUM_TEST = 1000
qp.environ["_R_SEED"] = 0
CSV_SEP = ","

_valid_problems = ["binary", "multiclass"]
PROBLEM = "binary"
