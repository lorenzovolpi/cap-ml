import os

import cap_exp
from cap import env as capenv

PROJECT = "reverse"

cap_exp.env.load_env()
root_dir = os.path.join(capenv["OUT_DIR"], PROJECT)
