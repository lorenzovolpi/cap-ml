
from sklearn.linear_model import LogisticRegression
from quacc.baseline import kfcv
from quacc.dataset import get_spambase


class TestBaseline:

    def test_kfcv(self):
        train, _, _ = get_spambase()
        c_model = LogisticRegression()
        assert "f1_score" in kfcv(c_model, train)