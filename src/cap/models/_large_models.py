import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import quapy as qp
import torch
import torch.nn as nn
from quapy.data.base import LabelledCollection
from sklearn.base import BaseEstimator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    DistilBertForSequenceClassification,
    get_scheduler,
)

import cap
from cap.data.base import TorchDataset, TorchLabelledCollection


def softmax(logits: torch.Tensor) -> np.ndarray:
    return nn.functional.softmax(logits, dim=-1).numpy()


class BaseEstimatorAdapter(BaseEstimator):
    def __init__(self, V_hidden_states, U_hidden_states, V_logits, U_logits):
        self.hs = np.vstack([V_hidden_states, U_hidden_states])
        self.logits = np.vstack([V_logits, U_logits])

        hashes = self._hash(self.hs)
        self._dict = defaultdict(lambda: [])
        for i, hash in enumerate(hashes):
            self._dict[hash].append(i)

    def _hash(self, X):
        return np.around(np.abs(X).sum(axis=-1) * self.hs.shape[0])

    def predict_proba(self, X: np.ndarray):
        def f(data, hash):
            _ids = np.array(self._dict[hash])
            _m = self.hs[_ids, :]
            _eq_idx = np.nonzero((_m == data).all(axis=-1))[0][0]
            return _ids[_eq_idx]

        hashes = self._hash(X)
        logits_idx = np.vectorize(f, signature="(m),()->()")(X, hashes)
        _logits = self.logits[logits_idx, :]
        return softmax(_logits, axis=-1)

    def decision_function(self, X: np.ndarray):
        return self.predict_proba(X)


class LargeModel:
    def fit(self, train: LabelledCollection, dataset_name: str): ...

    def predict_proba(self, test) -> np.ndarray: ...

    def get_model_path(self, dataset_name):
        return os.path.join(cap.env["OUT_DIR"], "models", f"{self.name}_on_{dataset_name}")

    def predict_from_proba(self, proba: np.ndarray) -> np.ndarray:
        return np.argmax(proba, axis=-1)

    def predict(self, test) -> np.ndarray:
        return self.predict_from_proba(self.predict_proba(test))


class DistilBert(LargeModel):
    def __init__(self, learning_rate=1e-5, batch_size=64, epochs=3, seed=None):
        self.name = "distilbert-base-uncased"
        self.tokenizer_name = self.name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_epochs = epochs
        self.seed = qp.environ["_R_SEED"] if seed is None else seed

        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer, padding="max_length")

    def get_model_path(self, dataset_name):
        return os.path.join(cap.env["OUT_DIR"], "models", f"{self.name}_on_{dataset_name}.tar")

    def checkpoint(self, dataset_name):
        path = self.get_model_path(dataset_name)
        parent_dir = Path(path).parent
        os.makedirs(parent_dir, exist_ok=True)
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            path,
        )

    def _load_base_model(self):
        self.model: DistilBertForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(self.name)
        self.optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        self.epoch = 0

    def load_from_checkpoint(self, dataset_name):
        path = self.get_model_path(dataset_name)
        if os.path.exists(path):
            _checkpoint = torch.load(path)
            self.model.load_state_dict(_checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(_checkpoint["optimizer_state_dict"])
            self.epoch = _checkpoint["epoch"]

    def fit(self, train: TorchLabelledCollection, dataset_name: str, verbose=True):
        self._load_base_model()
        self.load_from_checkpoint(dataset_name)

        self.classes_ = train.classes_
        train_dl = DataLoader(
            TorchDataset.from_lc(train),
            batch_size=self.batch_size,
        )
        num_training_steps = self.num_epochs * len(train_dl)

        if self.num_epochs == self.epoch:
            return self

        lr_scheduler = get_scheduler(
            name="linear", optimizer=self.optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(device)

        if verbose:
            progress_bar = tqdm(range(num_training_steps))

        self.model.train()
        for epoch in range(self.num_epochs - self.epoch):
            self.epoch = epoch + 1
            for batch in train_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()

                self.optimizer.step()
                lr_scheduler.step()
                self.optimizer.zero_grad()
                if verbose:
                    progress_bar.update(1)

        self.checkpoint(dataset_name)

        return self

    def predict_proba(self, test, attention_mask, verbose=False) -> np.ndarray:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        test_dl = DataLoader(
            TorchDataset.from_X(test, attention_mask),
            batch_size=self.batch_size,
        )
        self.model.to(device)

        self.model.eval()
        y_probs = []
        if verbose:
            progress_bar = tqdm(range(len(test_dl)))
        with torch.no_grad():
            for batch in test_dl:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(**batch)
                y_probs.append(softmax(outputs.logits.cpu()))
                if verbose:
                    progress_bar.update(1)

        return np.vstack(y_probs)
