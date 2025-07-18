import os
import sys
from pathlib import Path

if __name__ == "__main__":
    out_path = Path(sys.argv[1])
    if not (os.path.exists(out_path) and os.path.isfile(out_path)):
        print(out_path, "does not exist or is not a file", file=sys.stderr)
        exit(1)

    res = ""
    with open(out_path, "r") as f:
        for line in f.readlines():
            if (line[3:].startswith("%|") and line[:3] != "100") or (
                line[8:].startswith("%|") and line[:8] != "Map: 100"
            ):
                end_idx = line.find("]")
                line = line[end_idx + 1 :]
            if line.startswith("[warning] "):
                continue
            if len(line.strip()) == 0:
                continue
            res += line

    print(res)
