import os
import pickle
from glob import glob

from cap_exp.pretrain.data import PretrainInfo


def main():
    for path in glob(os.path.expanduser("~/tms/output/tms/pretrain/classic/*.pkl")):
        with open(path, "rb") as f:
            p_info: PretrainInfo = pickle.load(f)
            print(f"{p_info.d_info.name} - {p_info.h_info.name}: {p_info.h_info.default}")


if __name__ == "__main__":
    main()
