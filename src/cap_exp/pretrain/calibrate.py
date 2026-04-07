from argparse import ArgumentParser
from dataclasses import dataclass

import numpy as np
import quapy as qp
import scipy.special as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data import NotPretrainedError, PretrainInfo, load_info_paths
from env import PROJECT
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold
from tqdm import tqdm
from util import get_logger

EXPERIMENT = "calibrate"

qp.environ["SAMPLE_SIZE"] = 1000
qp.environ["_R_SEED"] = 0


@dataclass
class Calibrated:
    p: PretrainInfo
    posteriors: dict = None
    logits: dict = None
    exists: bool = False
    ece_pre: float = 0.0
    ece_post: float = 0.0


def _ece_bin(y, post, n_bins=10):
    prob_true, prob_pred = calibration_curve(y, post, n_bins=n_bins)
    return np.mean(np.abs(prob_pred - prob_true))


def _ece_multi(y, post, n_bins=10):
    pass
    confidences = post.max(axis=1)
    predictions = post.argmax(axis=1)
    accuracies = (predictions == y).astype(float)

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges[1:-1])

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = accuracies[mask].mean()
            bin_conf = confidences[mask].mean()
            weight = mask.sum() / len(y)
            ece += weight * abs(bin_acc - bin_conf)

    return ece


def ece(y, post, n_bins=10):
    if post.shape[-1] == 2:
        return _ece_bin(y, post[:, 1], n_bins)
    else:
        return _ece_multi(y, post, n_bins)


class TemperatureScaling(nn.Module):
    def __init__(self, n_splits=5, lr=0.01, max_iter=50):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self.n_splits = n_splits
        self.lr = lr
        self.max_iter = max_iter

    def forward(self, logits):
        return logits / self.temperature

    def _fit_fold(self, val_logits: torch.tensor, val_labels: torch.tensor):
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        with torch.no_grad():
            self.temperature.clamp_(min=0.01)

        optimal_T = self.temperature.item()
        return optimal_T

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray):
        if self.n_splits <= 1:
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)
            val_logits_pt = torch.tensor(val_logits)
            val_labels_pt = torch.tensor(val_labels)
            T = self._fit_fold(val_logits_pt, val_labels_pt)
            val_probs = self.predict_proba(val_logits)
            return T, ece(val_labels, val_probs)

        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=qp.environ["_R_SEED"])

        temperatures = []
        ece_scores = []

        for train_index, test_index in kf.split(val_logits):
            train_logits = torch.tensor(val_logits[train_index])
            train_labels = torch.tensor(val_labels[train_index])
            test_logits = val_logits[test_index]
            test_labels = val_labels[test_index]

            self.temperature = nn.Parameter(torch.ones(1) * 1.5)
            self._fit_fold(train_logits, train_labels)
            temperatures.append(self.temperature.item())

            test_probs = self.predict_proba(test_logits)
            ece_scores.append(ece(test_labels, test_probs))

        mean_T = np.mean(temperatures)
        # mean_ece = np.mean(ece_scores)

        self.temperature = nn.Parameter(torch.ones(1) * mean_T)

        val_probs = self.predict_proba(val_logits)
        return mean_T, ece(val_labels, val_probs)

    def predict_proba(self, test_logits: np.ndarray):
        _logits = torch.tensor(test_logits)
        with torch.no_grad():
            scaled_logits = self.forward(_logits)
            probs = F.softmax(scaled_logits, dim=1)

        return probs.cpu().numpy()


class VectorScaling(nn.Module):
    def __init__(self, n_classes: int, lr=0.01, max_iter=50):
        super().__init__()
        self.n_classes = n_classes
        self.lr = lr
        self.max_iter = max_iter
        self.temperature = nn.Parameter(torch.ones(n_classes) * 1.5)

    def forward(self, logits):
        return logits / self.temperature

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray):
        self.temperature = nn.Parameter(torch.ones(self.n_classes) * 1.5)
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=self.lr, max_iter=self.max_iter)

        val_logits_pt = torch.tensor(val_logits)
        val_labels_pt = torch.tensor(val_labels)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits_pt), val_labels_pt)
            loss.backward()
            return loss

        optimizer.step(eval_loss)

        with torch.no_grad():
            self.temperature.clamp_(min=0.01)

        optimal_T = self.temperature.detach().cpu().numpy()

        val_probs = self.predict_proba(val_logits)
        return optimal_T, ece(val_labels, val_probs)

    def predict_proba(self, test_logits: np.ndarray):
        _logits = torch.tensor(test_logits)
        with torch.no_grad():
            scaled_logits = self.forward(_logits)
            probs = F.softmax(scaled_logits, dim=1)

        return probs.cpu().numpy()


def calibrate(p: PretrainInfo, recalib=False):
    d_bundle = p.load_dataset_bundle()
    V_logits, U_logits = p.load_logits()

    _exist = True
    try:
        p.load_posteriors()
    except NotPretrainedError:
        _exist = False

    if _exist and not recalib:
        return Calibrated(p, exists=True)

    v_ece_pre = ece(d_bundle.V.y, sp.softmax(V_logits, axis=1))
    ts = TemperatureScaling(lr=3e-2)
    # ts = VectorScaling(p.d_info.n_classes)
    _, v_ece_post = ts.fit(V_logits, d_bundle.V.y)

    logits = dict(V=V_logits, U=U_logits)
    posteriors = dict(V=ts.predict_proba(V_logits), U=ts.predict_proba(U_logits))

    return Calibrated(p=p, posteriors=posteriors, logits=logits, ece_pre=v_ece_pre, ece_post=v_ece_post)


def main(log, pargs):
    info_paths = load_info_paths(pargs.domain)
    p_infos = [PretrainInfo.load(p, fast=True) for p in info_paths]

    for p in tqdm(p_infos, desc="Calibration"):
        c = calibrate(p, recalib=pargs.recalib)
        if c.exists:
            log.info(f"[{c.p.h_info.name}@{c.p.d_info.name}] already calibrated, skipping.")
        else:
            c.p.dump(posteriors=c.posteriors, logits=c.logits)
            log.info(f"[{c.p.h_info.name}@{c.p.d_info.name}] calibrated: {c.ece_pre} -> {c.ece_post}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--recalib", action="store_true", help="Recalibrate existing posteriors")
    parser.add_argument("--text", action="store_const", dest="domain", const="text")
    parser.add_argument("--image", action="store_const", dest="domain", const="image")
    parser.add_argument("--classic", action="store_const", dest="domain", const="classic")
    pargs = parser.parse_args()

    if pargs.domain is None:
        raise ValueError("Please specify a domain.")

    log = get_logger(id=f"{PROJECT}.{EXPERIMENT}.{pargs.domain}")

    log.info("-" * 31 + "  start  " + "-" * 31)
    main(log, pargs)
    log.info("-" * 32 + "  end  " + "-" * 32)
