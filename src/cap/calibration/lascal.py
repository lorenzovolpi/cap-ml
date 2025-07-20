from typing import override

import numpy as np
import torch
from lascal import Calibrator
from scipy.special import softmax

from cap.calibration.base import SourceTargetCalibratorFactory


def np2tensor(scores, probability_to_logit=False):
    scores = torch.from_numpy(scores)
    if probability_to_logit:
        scores = torch.log(scores)
    return scores


class LasCal(SourceTargetCalibratorFactory):
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

    @override
    def __call__(self, Zsrc, ysrc, Ztgt):
        return self.calibrate(Zsrc, ysrc, Ztgt)
