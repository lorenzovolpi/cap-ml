from quapy.data.datasets import UCI_BINARY_DATASETS

from quacc.data.datasets import fetch_UCIBinaryDataset

if __name__ == "__main__":
    for dn in UCI_BINARY_DATASETS:
        L, V, U = fetch_UCIBinaryDataset(dn)
        print(f"{dn}: L={len(L)}; V={len(V)}; U={len(U)}")
