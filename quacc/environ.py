from pathlib import Path

defalut_env = {
    "DATASET_NAME": "rcv1",
    "DATASET_TARGET": "CCAT",
    "COMP_ESTIMATORS": [
        "OUR_BIN_SLD",
        "OUR_MUL_SLD",
        "KFCV",
        "ATC_MC",
        "ATC_NE",
        "DOC_FEAT",
        # "RCA",
        # "RCA_STAR",
    ],
    "DATASET_N_PREVS": 9,
    "OUT_DIR": Path("out"),
    "PLOT_OUT_DIR": Path("out/plot"),
    "PROTOCOL_N_PREVS": 21,
    "PROTOCOL_REPEATS": 100,
    "SAMPLE_SIZE": 1000,
}


class Environ:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)


env = Environ(**defalut_env)
