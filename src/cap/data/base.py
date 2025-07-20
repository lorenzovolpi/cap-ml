import torch
from quapy.data.base import LabelledCollection
from scipy.sparse import issparse


class TorchLabelledCollection(LabelledCollection):
    def __init__(self, instances, labels, attention_mask, classes=None):
        self.attention_mask = attention_mask
        super().__init__(instances, labels, classes)

    def sampling_from_index(self, index):
        documents = self.instances[index]
        labels = self.labels[index]
        attention_mask = self.attention_mask[index]
        return TorchLabelledCollection(documents, labels, attention_mask, classes=self.classes_)


class TorchDataset(torch.utils.data.Dataset):
    DATA_MAPPING = {
        "X": "input_ids",
        "y": "labels",
        "mask": "attention_mask",
    }

    def __init__(self, X, y, attention_mask, data_mapping=None, has_labels=True):
        self.X = X
        self.y = y
        self.attention_mask = attention_mask
        self.data_mapping = self.DATA_MAPPING if data_mapping is None else data_mapping
        self.has_labels = has_labels

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        res = {}
        if issparse(self.X):
            res["X"] = self.X[index, :].toarray()
        else:
            res["X"] = self.X[index, :]

        if self.has_labels:
            res["y"] = self.y[index]

        res["mask"] = self.attention_mask[index, :]

        return {self.data_mapping[k]: v for k, v in res.items()}

    @classmethod
    def from_lc(cls, data: TorchLabelledCollection, data_mapping=None):
        return TorchDataset(*data.Xy, data.attention_mask, data_mapping=data_mapping)

    @classmethod
    def from_X(cls, data, attention_mask, data_mapping=None):
        return TorchDataset(data, None, attention_mask, data_mapping=data_mapping, has_labels=False)
