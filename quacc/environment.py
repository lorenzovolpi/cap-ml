import os

env = {
    "OUT_DIR": os.getenv("QUACC_OUT_DIR", "."),
    "QUACC_DATA": os.getenv("QUACC_DATA", os.path.expanduser("~/quacc_data")),
    "QUAPY_DATA": os.getenv("QUACC_QUAPY_DATA", os.path.expanduser("~/quapy_data")),
    "SKLEARN_DATA": os.getenv("QUACC_SKLEARN_DATA", os.path.expanduser("~/scikit_learn_data")),
    "N_JOBS": int(os.getenv("QUACC_N_JOBS", -2)),
    "FORCE_NJOBS": int(os.getenv("QUACC_FORCE_NJOBS", 0)) > 0,
}
