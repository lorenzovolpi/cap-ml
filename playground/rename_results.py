import os
from argparse import ArgumentParser as AP
from glob import glob

import quacc as qc

BASEDIR = "results"

match_path = "**/PrediQuant*.json"
replacements = {
    "(SLD)": "(SLD-ae)",
    "(KDEy)": "(KDEy-ae)",
}


def get_path(w_path):
    return os.path.join(qc.env["OUT_DIR"], BASEDIR, w_path)


def parse_args():
    parser = AP()
    parser.add_argument("--dry", action="store_false", dest="RENAME")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    paths = glob(get_path(match_path), recursive=True)
    for f in paths:
        for r, s in replacements.items():
            if f.find(r) > 0:
                nf = f.replace(r, s)
                print(f"{f} -> {nf}")
                if args.RENAME:
                    os.rename(f, nf)
