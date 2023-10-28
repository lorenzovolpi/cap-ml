import yaml

defalut_env = {
    "DATASET_NAME": "rcv1",
    "DATASET_TARGET": "CCAT",
    "METRICS": ["acc", "f1"],
    "COMP_ESTIMATORS": [],
    "PLOT_ESTIMATORS": [],
    "PLOT_STDEV": False,
    "DATASET_N_PREVS": 9,
    "DATASET_PREVS": None,
    "OUT_DIR_NAME": "output",
    "OUT_DIR": None,
    "PLOT_DIR_NAME": "plot",
    "PLOT_OUT_DIR": None,
    "DATASET_DIR_UPDATE": False,
    "PROTOCOL_N_PREVS": 21,
    "PROTOCOL_REPEATS": 100,
    "SAMPLE_SIZE": 1000,
}


class environ:
    _instance = None

    def __init__(self, **kwargs):
        self.exec = []
        self.confs = []
        self._default = kwargs
        self.__setdict(kwargs)
        self.load_conf()

    def __setdict(self, d):
        for k, v in d.items():
            self.__setattr__(k, v)
        if len(self.PLOT_ESTIMATORS) == 0:
            self.PLOT_ESTIMATORS = self.COMP_ESTIMATORS

    def __class_getitem__(cls, k):
        env = cls.get()
        return env.__getattribute__(k)

    def load_conf(self):
        with open("conf.yaml", "r") as f:
            confs = yaml.safe_load(f)["exec"]

        _global = confs["global"]
        _estimators = set()
        for pc in confs["plot_confs"].values():
            _estimators = _estimators.union(set(pc["PLOT_ESTIMATORS"]))
        _global["COMP_ESTIMATORS"] = list(_estimators)

        self.plot_confs = confs["plot_confs"]

        for dataset in confs["datasets"]:
            self.confs.append(_global | dataset)

    def get_confs(self):
        for _conf in self.confs:
            self.__setdict(self._default)
            self.__setdict(_conf)
            if "DATASET_TARGET" not in _conf:
                self.DATASET_TARGET = None

            name = self.DATASET_NAME
            if self.DATASET_TARGET is not None:
                name += f"_{self.DATASET_TARGET}"
            name += f"_{self.DATASET_N_PREVS}prevs"

            yield name

    def get_plot_confs(self):
        for k, pc in self.plot_confs.items():
            if "PLOT_ESTIMATORS" in pc:
                self.PLOT_ESTIMATORS = pc["PLOT_ESTIMATORS"]
            if "PLOT_STDEV" in pc:
                self.PLOT_STDEV = pc["PLOT_STDEV"]

            name = self.DATASET_NAME
            if self.DATASET_TARGET is not None:
                name += f"_{self.DATASET_TARGET}"
            name += f"_{k}"
            yield name


env = environ(**defalut_env)
