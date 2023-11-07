import itertools
from copy import deepcopy
from time import time
from typing import Callable, Union
import numpy as np

import quapy as qp
from quapy.data import LabelledCollection
from quapy.model_selection import GridSearchQ
from quapy.protocol import UPP, AbstractProtocol, OnLabelledCollectionProtocol
from sklearn.base import BaseEstimator

import quacc as qc
import quacc.error
from quacc.data import ExtendedCollection
from quacc.evaluation import evaluate
from quacc.logger import SubLogger
from quacc.method.base import (
    BaseAccuracyEstimator,
    BinaryQuantifierAccuracyEstimator,
    MultiClassAccuracyEstimator,
)


class GridSearchAE(BaseAccuracyEstimator):
    def __init__(
        self,
        model: BaseAccuracyEstimator,
        param_grid: dict,
        protocol: AbstractProtocol,
        error: Union[Callable, str] = qc.error.maccd,
        refit=True,
        # timeout=-1,
        # n_jobs=None,
        verbose=False,
    ):
        self.model = model
        self.param_grid = self.__normalize_params(param_grid)
        self.protocol = protocol
        self.refit = refit
        # self.timeout = timeout
        # self.n_jobs = qp._get_njobs(n_jobs)
        self.verbose = verbose
        self.__check_error(error)
        assert isinstance(protocol, AbstractProtocol), "unknown protocol"

    def _sout(self, msg):
        if self.verbose:
            print(f"[{self.__class__.__name__}]: {msg}")

    def __normalize_params(self, params):
        __remap = {}
        for key in params.keys():
            k, delim, sub_key = key.partition("__")
            if delim and k == "q":
                __remap[key] = f"quantifier__{sub_key}"

        return {(__remap[k] if k in __remap else k): v for k, v in params.items()}

    def __check_error(self, error):
        if error in qc.error.ACCURACY_ERROR:
            self.error = error
        elif isinstance(error, str):
            self.error = qc.error.from_name(error)
        elif hasattr(error, "__call__"):
            self.error = error
        else:
            raise ValueError(
                f"unexpected error type; must either be a callable function or a str representing\n"
                f"the name of an error function in {qc.error.ACCURACY_ERROR_NAMES}"
            )

    def fit(self, training: LabelledCollection):
        """Learning routine. Fits methods with all combinations of hyperparameters and selects the one minimizing
            the error metric.

        :param training: the training set on which to optimize the hyperparameters
        :return: self
        """
        params_keys = list(self.param_grid.keys())
        params_values = list(self.param_grid.values())

        protocol = self.protocol

        self.param_scores_ = {}
        self.best_score_ = None

        tinit = time()

        hyper = [
            dict(zip(params_keys, val)) for val in itertools.product(*params_values)
        ]

        # self._sout(f"starting model selection with {self.n_jobs =}")
        self._sout("starting model selection")

        scores = [self.__params_eval(params, training) for params in hyper]

        for params, score, model in scores:
            if score is not None:
                if self.best_score_ is None or score < self.best_score_:
                    self.best_score_ = score
                    self.best_params_ = params
                    self.best_model_ = model
                self.param_scores_[str(params)] = score
            else:
                self.param_scores_[str(params)] = "timeout"

        tend = time() - tinit

        if self.best_score_ is None:
            raise TimeoutError("no combination of hyperparameters seem to work")

        self._sout(
            f"optimization finished: best params {self.best_params_} (score={self.best_score_:.5f}) "
            f"[took {tend:.4f}s]"
        )
        log = SubLogger.logger()
        log.debug(
            f"[{self.model.__class__.__name__}] "
            f"optimization finished: best params {self.best_params_} (score={self.best_score_:.5f}) "
            f"[took {tend:.4f}s]"
        )

        if self.refit:
            if isinstance(protocol, OnLabelledCollectionProtocol):
                self._sout("refitting on the whole development set")
                self.best_model_.fit(training + protocol.get_labelled_collection())
            else:
                raise RuntimeWarning(
                    f'"refit" was requested, but the protocol does not '
                    f"implement the {OnLabelledCollectionProtocol.__name__} interface"
                )

        return self

    def __params_eval(self, params, training):
        protocol = self.protocol
        error = self.error

        # if self.timeout > 0:

        #     def handler(signum, frame):
        #         raise TimeoutError()

        #     signal.signal(signal.SIGALRM, handler)

        tinit = time()

        # if self.timeout > 0:
        #     signal.alarm(self.timeout)

        try:
            model = deepcopy(self.model)
            # overrides default parameters with the parameters being explored at this iteration
            model.set_params(**params)
            # print({k: v for k, v in model.get_params().items() if k in params})
            model.fit(training)
            score = evaluate(model, protocol=protocol, error_metric=error)

            ttime = time() - tinit
            self._sout(
                f"hyperparams={params}\t got score {score:.5f} [took {ttime:.4f}s]"
            )

            # if self.timeout > 0:
            #     signal.alarm(0)
        # except TimeoutError:
        #     self._sout(f"timeout ({self.timeout}s) reached for config {params}")
        #     score = None
        except ValueError as e:
            self._sout(f"the combination of hyperparameters {params} is invalid")
            raise e
        except Exception as e:
            self._sout(f"something went wrong for config {params}; skipping:")
            self._sout(f"\tException: {e}")
            score = None

        return params, score, model

    def extend(self, coll: LabelledCollection, pred_proba=None) -> ExtendedCollection:
        assert hasattr(self, "best_model_"), "quantify called before fit"
        return self.best_model().extend(coll, pred_proba=pred_proba)

    def estimate(self, instances, ext=False):
        """Estimate class prevalence values using the best model found after calling the :meth:`fit` method.

        :param instances: sample contanining the instances
        :return: a ndarray of shape `(n_classes)` with class prevalence estimates as according to the best model found
            by the model selection process.
        """

        assert hasattr(self, "best_model_"), "estimate called before fit"
        return self.best_model().estimate(instances, ext=ext)

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



class MCAEgsq(MultiClassAccuracyEstimator):
    def __init__(
        self,
        classifier: BaseEstimator,
        quantifier: BaseAccuracyEstimator,
        param_grid: dict,
        error: Union[Callable, str] = qp.error.mae,
        refit=True,
        timeout=-1,
        n_jobs=None,
        verbose=False,
    ):
        self.param_grid = param_grid
        self.refit = refit
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.error = error
        super().__init__(classifier, quantifier)

    def fit(self, train: LabelledCollection):
        self.e_train = self.extend(train)
        t_train, t_val = self.e_train.split_stratified(0.6, random_state=0)
        self.quantifier = GridSearchQ(
            deepcopy(self.quantifier),
            param_grid=self.param_grid,
            protocol=UPP(t_val, repeats=100),
            error=self.error,
            refit=self.refit,
            timeout=self.timeout,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
        ).fit(self.e_train)

        return self

    def estimate(self, instances, ext=False) -> np.ndarray:
        e_inst = instances if ext else self._extend_instances(instances)
        estim_prev = self.quantifier.quantify(e_inst)
        return self._check_prevalence_classes(estim_prev, self.quantifier.best_model().classes_)


class BQAEgsq(BinaryQuantifierAccuracyEstimator):
    def __init__(
        self,
        classifier: BaseEstimator,
        quantifier: BaseAccuracyEstimator,
        param_grid: dict,
        error: Union[Callable, str] = qp.error.mae,
        refit=True,
        timeout=-1,
        n_jobs=None,
        verbose=False,
    ):
        self.param_grid = param_grid
        self.refit = refit
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.error = error
        super().__init__(classifier=classifier, quantifier=quantifier)

    def fit(self, train: LabelledCollection):
        self.e_train = self.extend(train)

        self.n_classes = self.e_train.n_classes
        self.e_trains = self.e_train.split_by_pred()

        self.quantifiers = []
        for e_train in self.e_trains:
            t_train, t_val = e_train.split_stratified(0.6, random_state=0)
            quantifier = GridSearchQ(
                model=deepcopy(self.quantifier),
                param_grid=self.param_grid,
                protocol=UPP(t_val, repeats=100),
                error=self.error,
                refit=self.refit,
                timeout=self.timeout,
                n_jobs=self.n_jobs,
                verbose=self.verbose,
            ).fit(t_train)
            self.quantifiers.append(quantifier)

        return self
