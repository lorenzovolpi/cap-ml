import json
import os
import subprocess as sb

import cap

def load_env():
    if os.path.exists("env.json"):
        _hostname = sb.run(["hostname"], capture_output=True).stdout.decode("UTF-8").strip()
        with open("env.json", "r") as f:
            _jdict = json.load(f)
            _env = _hostname if _hostname in _jdict else "global"
            ext_env = _jdict.get(_env, cap.env)
        cap.env |= ext_env

_valid_problems = ["binary", "multiclass"]
