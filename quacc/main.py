import numpy as np
import quapy as qp
import scipy.sparse as sp
from quapy.data import LabelledCollection
from quapy.protocol import APP, AbstractStochasticSeededProtocol
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.svm import LinearSVC


# Extended classes
#
# 0 ~ True 0
# 1 ~ False 1
# 2 ~ False 0
# 3 ~ True 1
#      _____________________
#     |          |          |
#     |  True 0  |  False 1 |
#     |__________|__________|
#     |          |          |
#     |  False 0 |  True 1  |
#     |__________|__________|
#
def get_ex_class(classes, true_class, pred_class):
    return true_class * classes + pred_class


def extend_collection(coll, pred_prob):
    n_classes = coll.n_classes

    # n_X = [ X | predicted probs. ]
    if isinstance(coll.X, sp.csr_matrix):
        pred_prob_csr = sp.csr_matrix(pred_prob)
        n_x = sp.hstack([coll.X, pred_prob_csr])
    elif isinstance(coll.X, np.ndarray):
        n_x = np.concatenate((coll.X, pred_prob), axis=1)
    else:
        raise ValueError("Unsupported matrix format")

    # n_y = (exptected y, predicted y)
    n_y = []
    for i, true_class in enumerate(coll.y):
        pred_class = pred_prob[i].argmax(axis=0)
        n_y.append(get_ex_class(n_classes, true_class, pred_class))

    return LabelledCollection(n_x, np.asarray(n_y), [*range(0, n_classes * n_classes)])


def qf1e_binary(prev):
    recall = prev[0] / (prev[0] + prev[1])
    precision = prev[0] / (prev[0] + prev[2])

    return 1 - 2 * (precision * recall) / (precision + recall)


def compute_errors(true_prev, estim_prev, n_instances):
    errors = {}
    _eps = 1 / (2 * n_instances)
    errors = {
        "mae": qp.error.mae(true_prev, estim_prev),
        "rae": qp.error.rae(true_prev, estim_prev, eps=_eps),
        "mrae": qp.error.mrae(true_prev, estim_prev, eps=_eps),
        "kld": qp.error.kld(true_prev, estim_prev, eps=_eps),
        "nkld": qp.error.nkld(true_prev, estim_prev, eps=_eps),
        "true_f1e": qf1e_binary(true_prev),
        "estim_f1e": qf1e_binary(estim_prev),
    }

    return errors


def extend_and_quantify(
    model,
    q_model,
    train,
    test: LabelledCollection | AbstractStochasticSeededProtocol,
):
    model.fit(*train.Xy)

    pred_prob_train = cross_val_predict(model, *train.Xy, method="predict_proba")
    _train = extend_collection(train, pred_prob_train)

    q_model.fit(_train)

    def quantify_extended(test):
        pred_prob_test = model.predict_proba(test.X)
        _test = extend_collection(test, pred_prob_test)
        _estim_prev = q_model.quantify(_test.instances)
        # check that _estim_prev has all the classes and eventually fill the missing 
        # ones with 0
        for _cls in _test.classes_:
            if _cls not in q_model.classes_:
                _estim_prev = np.insert(_estim_prev, _cls, [0.0], axis=0)
                print(_estim_prev)
        return _test.prevalence(), _estim_prev

    if isinstance(test, LabelledCollection):
        _orig_prev, _true_prev, _estim_prev = quantify_extended(test)
        _errors = compute_errors(_true_prev, _estim_prev, test.X.shape[0])
        return ([_orig_prev], [_true_prev], [_estim_prev], [_errors])

    elif isinstance(test, AbstractStochasticSeededProtocol):
        orig_prevs, true_prevs, estim_prevs, errors = [], [], [], []
        for index in test.samples_parameters():
            sample = test.sample(index)
            _true_prev, _estim_prev = quantify_extended(sample)

            orig_prevs.append(sample.prevalence())
            true_prevs.append(_true_prev)
            estim_prevs.append(_estim_prev)
            errors.append(compute_errors(_true_prev, _estim_prev, sample.X.shape[0]))

        return orig_prevs, true_prevs, estim_prevs, errors


def get_dataset(name):
    datasets = {
        "spambase": lambda: qp.datasets.fetch_UCIDataset(
            "spambase", verbose=False
        ).train_test,
        "hp": lambda: qp.datasets.fetch_reviews("hp", tfidf=True).train_test,
        "imdb": lambda: qp.datasets.fetch_reviews("imdb", tfidf=True).train_test,
    }

    try:
        return datasets[name]()
    except KeyError:
        raise KeyError(f"{name} is not available as a dataset")


def test_1(dataset_name):
    train, test = get_dataset(dataset_name)

    orig_prevs, true_prevs, estim_prevs, errors = extend_and_quantify(
        LogisticRegression(),
        qp.method.aggregative.SLD(LogisticRegression()),
        train,
        APP(test, sample_size=100, n_prevalences=11, repeats=1),
    )

    for orig_prev, true_prev, estim_prev, _errors in zip(
        orig_prevs, true_prevs, estim_prevs, errors
    ):
        print(f"original prevalence:\t{orig_prev}")
        print(f"true prevalence:\t{true_prev}")
        print(f"estimated prevalence:\t{estim_prev}")
        for name, err in _errors.items():
            print(f"{name}={err:.3f}")
        print()


def main():
    for dataset_name in [
        # "hp",
        # "imdb",
        "spambase",
    ]:
        print(dataset_name)
        test_1(dataset_name)
        print("*" * 50)


if __name__ == "__main__":
    main()
