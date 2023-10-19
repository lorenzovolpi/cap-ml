
from sklearn.linear_model import LogisticRegression
from quacc.evaluation.baseline import kfcv, trust_score
from quacc.dataset import get_spambase


class TestBaseline:

    def test_kfcv(self):
        train, validation, _ = get_spambase()
        c_model = LogisticRegression()
        c_model.fit(train.X, train.y)
        assert "f1_score" in kfcv(c_model, validation)

    def test_trust_score(self):
        train, validation, test = get_spambase()
        c_model = LogisticRegression()
        c_model.fit(train.X, train.y)
        trustscore = trust_score(c_model, train, test)
        assert len(trustscore) == len(test.y)