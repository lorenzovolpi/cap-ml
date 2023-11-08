from sklearn.linear_model import LogisticRegression

from quacc.dataset import Dataset
from quacc.evaluation.baseline import kfcv


class TestBaseline:
    def test_kfcv(self):
        spambase = Dataset("spambase", n_prevalences=1).get_raw()
        c_model = LogisticRegression()
        c_model.fit(spambase.train.X, spambase.train.y)
        assert "f1_score" in kfcv(c_model, spambase.validation)
