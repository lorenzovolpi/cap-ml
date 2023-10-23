import yaml

defalut_env = {
    "DATASET_NAME": "rcv1",
    "DATASET_TARGET": "CCAT",
    "METRICS": ["acc", "f1"],
    "COMP_ESTIMATORS": [
        "our_bin_SLD",
        "our_bin_SLD_nbvs",
        "our_bin_SLD_bcts",
        "our_bin_SLD_ts",
        "our_bin_SLD_vs",
        "our_bin_CC",
        "our_mul_SLD",
        "our_mul_SLD_nbvs",
        "our_mul_SLD_bcts",
        "our_mul_SLD_ts",
        "our_mul_SLD_vs",
        "our_mul_CC",
        "ref",
        "kfcv",
        "atc_mc",
        "atc_ne",
        "doc_feat",
        "rca",
        "rca_star",
    ],
    "DATASET_N_PREVS": 9,
    "OUT_DIR_NAME": "output",
    "PLOT_DIR_NAME": "plot",
    "PROTOCOL_N_PREVS": 21,
    "PROTOCOL_REPEATS": 100,
    "SAMPLE_SIZE": 1000,
}


class Environ:
    def __init__(self, **kwargs):
        self.exec = []
        self.confs = {}
        self.__setdict(kwargs)

    def __setdict(self, d):
        for k, v in d.items():
            self.__setattr__(k, v)

    def load_conf(self):
        with open("conf.yaml", "r") as f:
            confs = yaml.safe_load(f)

        for common in confs["commons"]:
            name = common["DATASET_NAME"]
            if "DATASET_TARGET" in common:
                name += "_" + common["DATASET_TARGET"]
            for k, d in confs["confs"].items():
                _k = f"{name}_{k}"
                self.confs[_k] = common | d
                self.exec.append(_k)

        if "exec" in confs:
            if len(confs["exec"]) > 0:
                self.exec = confs["exec"]

    def __iter__(self):
        self.load_conf()
        for _conf in self.exec:
            if _conf in self.confs:
                self.__setdict(self.confs[_conf])
                yield _conf


env = Environ(**defalut_env)
