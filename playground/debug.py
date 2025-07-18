import numpy as np
import quapy as qp
from quapy.method.aggregative import EMQ
from quapy.protocol import UPP
from sklearn.linear_model import LogisticRegression as LR

from quacc.data.datasets import fetch_RCV1BinaryDataset, fetch_RCV1MulticlassDataset
from quacc.error import vanilla_acc
from quacc.experiments.generators import gen_acc_measure
from quacc.experiments.util import fit_or_switch, get_predictions, prevs_from_prot, split_validation
from quacc.models.cont_table import QuAcc1xN2, QuAccNxN
from quacc.models.model_selection import GridSearchCAP as GSCAP
from quacc.utils.commons import true_acc


def sld():
    return EMQ(LR(), val_split=5)


ORACLE = False
NUM_TEST = 1000
R_SEED = 42
qp.environ["_R_SEED"] = R_SEED
qp.environ["SAMPLE_SIZE"] = 100

pacc_lr_params = {
    "q_class__classifier__C": np.logspace(-3, 3, 7),
    "q_class__classifier__class_weight": [None, "balanced"],
    "add_posteriors": [True, False],
    "add_y_hat": [True, False],
    "add_maxconf": [True, False],
    "add_negentropy": [True, False],
    "add_maxinfsoft": [True, False],
}
emq_lr_params = pacc_lr_params | {"q_class__recalib": [None, "bcts"]}
kde_lr_params = pacc_lr_params | {"q_class__bandwidth": np.linspace(0.01, 0.2, 5)}

if __name__ == "__main__":
    L, V, U = fetch_RCV1MulticlassDataset("C1")
    h = LR()
    h.fit(*L.Xy)

    test_prot = UPP(U, repeats=NUM_TEST, return_type="labelled_collection", random_state=0)
    V1, V2_prot = split_validation(V)

    V_posteriors = h.predict_proba(V.X)
    V1_posteriors = h.predict_proba(V1.X)
    V2_prot_posteriors = []
    for sample in V2_prot():
        V2_prot_posteriors.append(h.predict_proba(sample.X))

    test_prot_posteriors, test_prot_y_hat = [], []
    for sample in test_prot():
        P = h.predict_proba(sample.X)
        test_prot_posteriors.append(P)
        test_prot_y_hat.append(np.argmax(P, axis=-1))

    true_accs = {}
    for acc_name, acc_fn in gen_acc_measure(multiclass=True):
        true_accs[acc_name] = [true_acc(h, acc_fn, Ui) for Ui in test_prot()]

    methods = [
        (
            "QuAcc(SLD)1xn2-OPT",
            GSCAP(
                QuAcc1xN2(acc_fn, sld()),
                emq_lr_params,
                V2_prot,
                V2_prot_posteriors,
                acc_fn,
                refit=False,
                # n_jobs=0,
                raise_errors=True,
            ),
            V1,
            V1_posteriors,
        ),
        (
            "QuAcc(KDEy)1x2-OPT",
            GSCAP(
                QuAcc1xN2(acc_fn, sld()),
                emq_lr_params,
                V2_prot,
                V2_prot_posteriors,
                acc_fn,
                refit=False,
                # n_jobs=0,
                raise_errors=True,
            ),
            V1,
            V1_posteriors,
        ),
    ]

    for method_name, method, val, val_posteriors in methods:
        acc_name, acc_fn = "vanilla_accuracy", vanilla_acc
        method, t_train = fit_or_switch(method, val, val_posteriors, acc_fn, False)
        estim_accs, t_test_ave = get_predictions(method, test_prot, test_prot_posteriors, ORACLE)
        estim_accs = np.asarray(estim_accs)
        print(f"{method_name}:\t{estim_accs.mean()} [{t_train}s]")
