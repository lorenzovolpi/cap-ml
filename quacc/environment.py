import collections as C
import copy
from typing import Any

import yaml


class environ:
    _instance = None
    _default_env = {
        "DATASET_NAME": None,
        "DATASET_TARGET": None,
        "METRICS": [],
        "COMP_ESTIMATORS": [],
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
        "PLOT_ESTIMATORS": [],
        "PLOT_STDEV": False,
    }
    _keys = list(_default_env.keys())

    def __init__(self):
        self.exec = []
        self.confs = []
        self.load_conf()
        self._stack = C.deque([self.__getdict()])

    def __setdict(self, d):
        for k, v in d.items():
            super().__setattr__(k, v)

    def __getdict(self):
        return {k: self.__getattribute__(k) for k in environ._keys}

    def __setattr__(self, __name: str, __value: Any) -> None:
        if __name in environ._keys:
            self._stack[-1][__name] = __value
        super().__setattr__(__name, __value)

    def load_conf(self):
        self.__setdict(environ._default_env)

        with open("conf.yaml", "r") as f:
            confs = yaml.safe_load(f)["exec"]

        _global = confs["global"]
        _estimators = set()
        for pc in confs["plot_confs"].values():
            _estimators = _estimators.union(set(pc["PLOT_ESTIMATORS"]))
        _global["COMP_ESTIMATORS"] = list(_estimators)

        self.__setdict(_global)

        self.confs = confs["confs"]
        self.plot_confs = confs["plot_confs"]

    def get_confs(self):
        self._stack.append(None)
        for _conf in self.confs:
            self._stack.pop()
            self.__setdict(self._stack[-1])
            self.__setdict(_conf)
            self._stack.append(self.__getdict())

            yield copy.deepcopy(self._stack[-1])

        self._stack.pop()

    def get_plot_confs(self):
        self._stack.append(None)
        for k, pc in self.plot_confs.items():
            self._stack.pop()
            self.__setdict(self._stack[-1])
            self.__setdict(pc)
            self._stack.append(self.__getdict())

            name = self.DATASET_NAME
            if self.DATASET_TARGET is not None:
                name += f"_{self.DATASET_TARGET}"
            name += f"_{k}"
            yield name

        self._stack.pop()

    @property
    def current(self):
        return copy.deepcopy(self.__getdict())


env = environ()

if __name__ == "__main__":
    stack = C.deque()
    stack.append(-1)

    def __gen(stack: C.deque):
        stack.append(None)
        for i in range(5):
            stack.pop()
            stack.append(i)
            yield stack[-1]

        stack.pop()

    print(stack)

    for i in __gen(stack):
        print(stack, i)

    print(stack)
