import itertools as IT
from copy import deepcopy
from enum import Enum
from time import time
from typing import Callable

import numpy as np
import quapy as qp
from quapy.protocol import AbstractProtocol, OnLabelledCollectionProtocol
from quapy.util import timeout

import cap
from cap.models.cont_table import CAPContingencyTable, CAPContingencyTableQ, LabelledCollection


class Status(Enum):
    SUCCESS = 1
    TIMEOUT = 2
    INVALID = 3
    ERROR = 4


class ConfigStatus:
    def __init__(self, params, status, msg=""):
        self.params = params
        self.status = status
        self.msg = msg

    def __str__(self):
        return f":params:{self.params} :status:{self.status} " + self.msg

    def __repr__(self):
        return str(self)

    def success(self):
        return self.status == Status.SUCCESS

    def failed(self):
        return self.status != Status.SUCCESS


class GridSearchCAP(CAPContingencyTable):
    def __init__(
        self,
        model: CAPContingencyTableQ,
        param_grid: dict,
        protocol: AbstractProtocol,
        prot_posteriors,
        acc_fn: Callable | str = cap.error.vanilla_acc,
        error: Callable | str = cap.error.mae,
        refit=True,
        timeout=-1,
        n_jobs=None,
        raise_errors=False,
        verbose=False,
    ):
        self.model = model
        self.param_grid = param_grid
        self.protocol = protocol
        self.prot_posteriors = prot_posteriors
        self.refit = refit
        self.timeout = timeout
        self.n_jobs = cap.commons.get_njobs(n_jobs)
        self.raise_errors = raise_errors
        self.verbose = verbose
        self.__check_acc_fn(acc_fn)
        self.__check_error(error)
        assert isinstance(protocol, AbstractProtocol), "unknown protocol"

    def _sout(self, msg):
        if self.verbose:
            print(f"[{self.__class__.__name__}:{self.model.__class__.__name__}]: {msg}")

    def __check_acc_fn(self, acc_fn):
        if acc_fn in cap.error.ACCURACY_MEASURE:
            self.acc_fn = acc_fn
        elif isinstance(acc_fn, str):
            self.acc_fn = cap.error.from_name(acc_fn)
        elif hasattr(acc_fn, "__call__"):
            self.acc_fn = acc_fn
        else:
            raise ValueError(
                f"unexpected accuracy function type; must either be a callable function or a str\n"
                f"representing the name of an error function in {cap.error.ACCURACY_MEASURE_NAMES}"
            )

    def __check_error(self, error):
        if error in cap.error.ACCURACY_ERROR:
            self.error = error
        elif isinstance(error, str) and error in cap.error.ACCURACY_ERROR_NAMES:
            self.error = cap.error.from_name(error)
        elif hasattr(error, "__call__"):
            self.error = error
        else:
            raise ValueError(
                f"unexpected error type; must either be a callable function or a str\n"
                f"representing the name of an error function in {cap.error.ACCURACY_ERROR_NAMES}"
            )

    def _evaluate(self, model: CAPContingencyTableQ):
        estim_tables = [
            model.predict_ct(sample.X, P) for sample, P in IT.zip_longest(self.protocol(), self.prot_posteriors)
        ]
        estim_accs = np.array([self.acc_fn(cont_table) for cont_table in estim_tables]).reshape(-1, 1)

        return self.error(self.prot_true_accs, estim_accs)

    def _prepare_classifier(self, cls_params):
        model = deepcopy(self.model)

        def job(cls_params):
            model.set_params(**cls_params)
            data = model.preprocess_data(self._training, self._training_posteriors)
            model.prepare_quantifier()
            predictions = model.quant_classifier_fit_predict(data)
            return predictions, data

        output, status, took = self._error_handler(job, cls_params)
        predictions, data = (None, None) if output is None else output
        self._sout(f"[classifier fit] hyperparams={cls_params} [took {took:.3f}s]")
        return model, predictions, data, status, took

    def _prepare_aggregation(self, args):
        model, predictions, data, cls_took, cls_params, q_params = args
        model = deepcopy(model)
        params = {**cls_params, **q_params}

        def job(q_params):
            model.set_params(**q_params)
            model.quant_aggregation_fit(predictions, data)
            score = self._evaluate(model)
            return score

        score, status, aggr_took = self._error_handler(job, q_params)
        self._print_status(params, score, status, aggr_took)
        return model, params, score, status, (cls_took + aggr_took)

    def _compute_scores(self, training, training_posteriors):
        # break down the set of hyperparameters into two: classifier-specific, quantifier-specific
        cls_configs, q_configs = group_params(self.param_grid)

        # train all classifiers and get the predictions
        self._training = training
        self._training_posteriors = training_posteriors
        if self.n_jobs == 0:
            cls_outs = [self._prepare_classifier(cfg) for cfg in cls_configs]
        else:
            cls_outs = cap.commons.parallel(
                self._prepare_classifier,
                cls_configs,
                seed=qp.environ.get("_R_SEED", None),
                n_jobs=self.n_jobs,
                return_as="list",
            )

        # filter out classifier configurations that yielded any error
        success_outs = []
        for (model, predictions, data, status, took), cls_config in zip(cls_outs, cls_configs):
            if status.success():
                success_outs.append((model, predictions, data, took, cls_config))
            else:
                self.error_collector.append(status)

        if len(success_outs) == 0:
            raise ValueError("No valid configuration found for the classifier!")

        # compute the true accuracy values for the samples generated by self.protocol
        self.prot_true_accs = [
            self.model.true_acc(Ui, P, acc_fn=self.acc_fn)
            for Ui, P in IT.zip_longest(self.protocol(), self.prot_posteriors)
        ]
        self.prot_true_accs = np.array(self.prot_true_accs).reshape(-1, 1)

        # explore the quantifier-specific hyperparameters for each valid training configuration
        aggr_configs = [(*out, q_config) for out, q_config in IT.product(success_outs, q_configs)]
        if self.n_jobs == 0:
            aggr_outs = [self._prepare_aggregation(args) for args in aggr_configs]
        else:
            aggr_outs = cap.commons.parallel(
                self._prepare_aggregation,
                aggr_configs,
                seed=qp.environ.get("_R_SEED", None),
                n_jobs=self.n_jobs,
                return_as="array",
            )

        return aggr_outs

    def _print_status(self, params, score, status, took):
        if status.success():
            self._sout(f"hyperparams=[{params}]\t got {self.error.__name__} = {score:.5f} [took {took:.3f}s]")
        else:
            self._sout(f"error={status}")

    def switch(self, acc_fn):
        return GridSearchCAP(
            deepcopy(self.model).switch(acc_fn),
            self.param_grid,
            self.protocol,
            self.prot_posteriors,
            acc_fn,
            self.error,
            self.refit,
            self.timeout,
            self.n_jobs,
            self.raise_errors,
            self.verbose,
        )

    def fit(self, training: LabelledCollection, posteriors):
        """Learning routine. Fits methods with all combinations of hyperparameters and selects the one minimizing
            the error metric.

        :param training: the training set on which to optimize the hyperparameters
        :return: self
        """

        if self.refit and not isinstance(self.protocol, OnLabelledCollectionProtocol):
            raise RuntimeWarning(
                f'"refit" was requested, but the protocol does not implement '
                f"the {OnLabelledCollectionProtocol.__name__} interface"
            )

        tinit = time()

        self.error_collector = []

        self._sout(f"starting model selection with n_jobs={self.n_jobs}")
        results = self._compute_scores(training, posteriors)

        self.param_scores_ = {}
        self.best_score_ = None
        for model, params, score, status, took in results:
            if status.success():
                if self.best_score_ is None or score < self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params
                    self.best_model_ = model
                self.param_scores_[str(params)] = score
            else:
                self.param_scores_[str(params)] = status.status
                self.error_collector.append(status)

        tend = time() - tinit

        if self.best_score_ is None:
            raise ValueError("no combination of hyperparameters seemed to work")

        self._sout(
            f"optimization finished: best params {self.best_params_} (score={self.best_score_:.5f}) [took {tend:.4f}s]"
        )

        no_errors = len(self.error_collector)
        if no_errors > 0:
            self._sout(f"warning: {no_errors} errors found")
            for err in self.error_collector:
                self._sout(f"\t{str(err)}")

        if self.refit:
            if isinstance(self.protocol, OnLabelledCollectionProtocol):
                tinit = time()
                self._sout("refitting on the whole development set")
                self.best_model_.fit(training + self.protocol.get_labelled_collection())
                tend = time() - tinit
                self.refit_time_ = tend
            else:
                # already checked
                raise RuntimeWarning("the model cannot be refit on the whole dataset")

        return self

    def predict_ct(self, test, posteriors, oracle_prev=None):
        """Estimate class prevalence values using the best model found after calling the :meth:`fit` method.

        :param instances: sample contanining the instances
        :return: a ndarray of shape `(n_classes)` with class prevalence estimates as according to the best model found
            by the model selection process.
        """
        assert hasattr(self, "best_model_"), "predict_ct called before fit"
        return self.best_model().predict_ct(test, posteriors, oracle_prev=oracle_prev)

    def set_params(self, **parameters):
        """Sets the hyper-parameters to explore.

        :param parameters: a dictionary with keys the parameter names and values the list of values to explore
        """
        self.param_grid = parameters

    def get_params(self, deep=True):
        """Returns the dictionary of hyper-parameters to explore (`param_grid`)

        :param deep: Unused
        :return: the dictionary `param_grid`
        """
        return self.param_grid

    def best_model(self):
        """
        Returns the best model found after calling the :meth:`fit` method, i.e., the one trained on the combination
        of hyper-parameters that minimized the error function.

        :return: a trained quantifier
        """
        if hasattr(self, "best_model_"):
            return self.best_model_
        raise ValueError("best_model called before fit")

    def _error_handler(self, func, params):
        """
        Endorses one job with two returned values: the status, and the time of execution

        :param func: the function to be called
        :param params: parameters of the function
        :return: `tuple(out, status, time)` where `out` is the function output,
            `status` is an enum value from `Status`, and `time` is the time it
            took to complete the call
        """

        output = None

        def _handle(status, exception):
            if self.raise_errors:
                raise exception
            else:
                return ConfigStatus(params, status)

        try:
            with timeout(self.timeout):
                tinit = time()
                output = func(params)
                status = ConfigStatus(params, Status.SUCCESS)

        except TimeoutError as e:
            status = _handle(Status.TIMEOUT, e)

        except ValueError as e:
            status = _handle(Status.INVALID, e)

        except Exception as e:
            status = _handle(Status.ERROR, e)

        took = time() - tinit
        return output, status, took


def group_params(param_grid: dict):
    """
    Partitions a param_grid dictionary as two lists of configurations, one for the classifier-specific
    hyper-parameters, and another for que quantifier-specific hyper-parameters

    :param param_grid: dictionary with keys representing hyper-parameter names, and values representing the range
        to explore for that hyper-parameter
    :return: two expanded grids of configurations, one for the classifier, another for the quantifier
    """
    classifier_params, quantifier_params = {}, {}
    for key, values in param_grid.items():
        if key.startswith("q_class__classifier__") or key.startswith("add_") or key == "q_class__val_split":
            classifier_params[key] = values
        else:
            quantifier_params[key] = values

    classifier_configs = qp.model_selection.expand_grid(classifier_params)
    quantifier_configs = qp.model_selection.expand_grid(quantifier_params)

    return classifier_configs, quantifier_configs
