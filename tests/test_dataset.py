import pytest
from quacc.dataset import Rcv1Helper


@pytest.fixture
def rcv1_helper() -> Rcv1Helper:
    return Rcv1Helper()


class TestDataset:
    def test_rcv1_binary_datasets(self, rcv1_helper):
        count = 0
        for X, Y, name in rcv1_helper.rcv1_binary_datasets():
            count += 1
            print(X.shape)
            assert X.shape == (517978, 47236)
            assert Y.shape == (517978,)

        assert count == 37

    @pytest.mark.parametrize("label", ["CCAT", "GCAT", "M11"])
    def test_rcv1_binary_dataset_by_label(self, rcv1_helper, label):
        train, test = rcv1_helper.rcv1_binary_dataset_by_label(label)
        assert train.X.shape == (23149, 47236)
        assert train.y.shape == (23149,)
        assert test.X.shape == (781265, 47236)
        assert test.y.shape == (781265,)

        assert (
            dict(rcv1_helper.documents_per_class_rcv1())[label]
            == train.y.sum() + test.y.sum()
        )
