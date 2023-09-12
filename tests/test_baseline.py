
from sklearn.linear_model import LogisticRegression
from quacc.baseline import kfcv
from quacc.dataset import get_spambase_traintest


class TestBaseline:

    def test_kfcv(self):
        train, _ = get_spambase_traintest()
        c_model = LogisticRegression()
        assert "f1_score" in kfcv(c_model, train)