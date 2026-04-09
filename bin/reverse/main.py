from dataclasses import dataclass

from cap.models.base import CAP
from cap_exp.pretrain.data import PretrainInfo

EXPERIMENT = "main"


@dataclass()
class EXP:
    code: int
    p: PretrainInfo
    acc_name: str
    method_name: str
    df: RDF = None
    t_train: float = None
    t_test_ave: float = None
    err: Exception = None

    @classmethod
    def SUCCESS(cls, *args, **kwargs):
        return EXP(200, *args, **kwargs)

    @classmethod
    def EXISTS(cls, *args, **kwargs):
        return EXP(300, *args, **kwargs)

    @classmethod
    def ERROR(cls, e, *args, **kwargs):
        return EXP(400, *args, err=e, **kwargs)

    @property
    def ok(self):
        return self.code == 200

    @property
    def old(self):
        return self.code == 300

    def error(self):
        return self.code == 400


def exp_protocol(args: tuple[str, str, CAP]) -> EXP:
    # bundle_path, method_name, method, acc_name, acc_fn = args
    # clsf, D, method_name, method, val, val_posteriors = args
    info_path, method_name, method = args
    results = []

    p = PretrainInfo.load(info_path, fast=True)
    d_info, h_info = p.d_info, p.h_info

    all_exist = True
    for acc in get_acc_names():
        all_exist = all_exist and os.path.exists(
            local_path(p.domain, d_info.name, h_info.full_name, method_name, acc, experiment=EXPERIMENT)
        )
    if all_exist:
        results.append(EXP.EXISTS(p, "all", method_name))
        return results

    D = p.load_dataset_bundle()
    h = p.load_pretrained_classifier(D)

    D.get_posteriors(h)
    if isinstance(method, NeedsValidationProtocol):
        val, val_posteriors = D.V1, D.V1_posteriors
        method.set_validation_protocol(D.V2_prot, D.V2_prot_posteriors)
    else:
        val, val_posteriors = D.V, D.V_posteriors

    # fit MS method
    try:
        tinit = time()
        method.fit(val, val_posteriors, D.test_prot, D.test_prot_posteriors)
        t_train = time() - tinit
    except Exception as e:
        results.append(EXP.ERROR(e, p, "fit", method_name))
        return results

    L_prev = get_plain_prev(D.L_prevalence)
    val_prev = get_plain_prev(val.prevalence())
    df_len = D.test_prot.total()
    test_shift = get_shift(np.array([Ui.prevalence() for Ui in D.test_prot()]), D.L_prevalence).tolist()
    # tp_true_cts = [ct.ravel() for ct in D.test_prot_true_cts]
    tp_true_cts = D.test_prot_true_cts

    for acc_name, acc_fn in gen_acc_measure(d_info.n_classes > 2):
        path = local_path(p.domain, d_info.name, h_info.full_name, method_name, acc_name, experiment=EXPERIMENT)
        if os.path.exists(path):
            results.append(EXP.EXISTS(p, acc_name, method_name))
            continue

        try:
            tinit = time()
            ranking_vals = method.rank(acc_fn)
            t_test_ave = (time() - tinit) / df_len
        except Exception as e:
            print_exception(e)
            results.append(EXP.ERROR(e, p, acc_name, method_name))
            continue

        # df_len = len(estim_accs)
        method_df = RDF.from_records(
            df_len,
            uids=np.arange(df_len).tolist(),
            shifts=test_shift,
            true_cts=tp_true_cts,
            ranking_vals=ranking_vals,
            classifier=h_info.name,
            classifier_class=h_info.class_name,
            default_c=[h_info.default] * df_len,
            ms_ignore=[h_info.ms_ignore] * df_len,
            method=method_name,
            dataset=d_info.name,
            collection=d_info.collection,
            n_classes=d_info.n_classes,
            acc_name=acc_name,
            train_prev=[L_prev] * df_len,
            val_prev=[val_prev] * df_len,
            t_train=t_train,
            t_test_ave=t_test_ave,
        )

        results.append(
            EXP.SUCCESS(
                p,
                acc_name,
                method_name,
                df=method_df,
                t_train=t_train,
                t_test_ave=t_test_ave,
            )
        )

    return results


def main():
    print("hello")


if __name__ == "__main__":
    main()
