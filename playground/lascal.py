from abc import ABC, abstractmethod

import numpy as np
import quapy as qp
import torch
from lascal import Calibrator
from quapy.protocol import UPP
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression

from quacc.data.datasets import fetch_UCIBinaryDataset

qp.environ["_R_SEED"] = 0


def np2tensor(scores, probability_to_logit=False):
    scores = torch.from_numpy(scores)
    if probability_to_logit:
        scores = torch.log(scores)
    return scores


class CalibratorSourceTarget(ABC):
    @abstractmethod
    def calibrate(self, Zsrc, ysrc, Ztgt): ...


class LasCalCalibration(CalibratorSourceTarget):
    def __init__(self, prob2logits=True):
        self.prob2logits = prob2logits

    def calibrate(self, Zsrc, ysrc, Ztgt):
        calibrator = Calibrator(
            experiment_path=None,
            verbose=False,
            covariate=False,
        )

        Zsrc = np2tensor(Zsrc, probability_to_logit=self.prob2logits)
        Ztgt = np2tensor(Ztgt, probability_to_logit=self.prob2logits)
        ysrc = np2tensor(ysrc)
        yte = None

        try:
            calibrated_agg = calibrator.calibrate(
                method_name="lascal",
                source_agg={"y_logits": Zsrc, "y_true": ysrc},
                target_agg={"y_logits": Ztgt, "y_true": yte},
                train_agg=None,
            )
            y_logits = calibrated_agg["target"]["y_logits"]
            Pte_calib = y_logits.softmax(-1).numpy()
        except Exception:
            Ztgt = Ztgt.numpy()
            if np.isclose(Ztgt.sum(axis=1), 1).all():
                Pte_calib = Ztgt
            else:
                Pte_calib = softmax(Ztgt, axis=1)

        return Pte_calib


if __name__ == "__main__":
    L, V, U = fetch_UCIBinaryDataset("spambase")
    h = LogisticRegression().fit(*L.Xy)

    V_post = h.predict_proba(V.X)

    test_prot = UPP(U, sample_size=100, repeats=100, random_state=0, return_type="labelled_collection")
    test_prot_post = [h.predict_proba(Ui.X) for Ui in test_prot()]

    calibrator = LasCalCalibration()
    test_prot_calib = [calibrator.calibrate(V_post, V.y, Ui_post) for Ui_post in test_prot_post]

    print(test_prot_calib)
