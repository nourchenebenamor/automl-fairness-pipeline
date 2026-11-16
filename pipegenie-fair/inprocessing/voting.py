# SPDX-License-Identifier: MIT-HUMAINS-Attribution
#
# Copyright (c) 2024 HUMAINS Research Group (University of Córdoba, Spain).
# Copyright (c) 2024 The authors.
# All rights reserved.
#
# MIT License – HUMAINS Research Group Attribution Variant
# For full license text, see the LICENSE file in the repository root.

"""
Soft voting / majority rule voting classifier and voting regressor.
"""

import warnings
from abc import ABC, abstractmethod
from itertools import repeat
from multiprocessing import cpu_count
from typing import TYPE_CHECKING

import numpy as np
from loky import ProcessPoolExecutor

from pipegenie._available_if import available_if
from pipegenie.metrics import accuracy_score, r2_score

from fairlearn.metrics import demographic_parity_difference

from fairlearn.metrics import MetricFrame
from pipegenie.metrics import root_mean_squared_error   

warnings.filterwarnings("ignore")

if TYPE_CHECKING:
    from typing import Optional

    from numpy.typing import ArrayLike, NDArray

# Ignore all warnings
warnings.filterwarnings("ignore")

# Show warnings from the "pipegenie" package
warnings.filterwarnings("default", module=r"^pipegenie\.")


class BaseVoting(ABC):
    """
    Base class for all voting ensembles.

    Ensemble methods combine the predictions of several base estimators
    in order to improve generalization / robustness over a single estimator.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The list of estimators to fit and vote on.

    weights : array-like of shape (n_estimators,) or None, default=None
        The weights of the estimators. If None, then uniform weights are used.

    n_jobs : int, default=1
        The number of jobs to run in parallel.
        If -1, then the number of jobs is set to the number of CPU cores.
    """

    def __init__(
        self,
        estimators: list[tuple[str, object]],
        weights: 'Optional[ArrayLike]' = None,
        n_jobs: int = 1,
    ):
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    @abstractmethod
    def fit(self, X: 'ArrayLike', y: 'ArrayLike') -> 'BaseVoting':
        """
        Fit the model to the data.
        """
        if self.weights is not None and len(self.weights) != len(self.estimators):
            raise ValueError("The number of weights should be equal to the number of estimators.")

        with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
            results = executor.map(
                self._fit_single_estimator,
                [est[1] for est in self.estimators],
                repeat(X),
                repeat(y),
            )

        self.estimators = list(zip([est[0] for est in self.estimators], results, strict=True))

        return self

    def _fit_single_estimator(self, estimator: object, X: 'ArrayLike', y: 'ArrayLike') -> object:
        return estimator.fit(X, y)

    @abstractmethod
    def predict(self, X: 'ArrayLike') -> 'NDArray':
        """
        Predict using the model.
        """
        raise NotImplementedError("predict method should be implemented by the subclass")

    @abstractmethod
    def score(self, X: 'ArrayLike', y: 'ArrayLike') -> float:
        """
        Return the score of the model on the input data.
        """
        raise NotImplementedError("score method should be implemented by the subclass")


class VotingClassifier(BaseVoting):
    """
    A voting classifier that uses the majority rule voting strategy.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The list of estimators to fit and vote on.

    weights : array-like of shape (n_classifiers,) or None, default=None
        The weights of the classifiers. If None, then uniform weights are used.

    n_jobs : int, default=1
        The number of jobs to run in parallel.
        If -1, then the number of jobs is set to the number of CPU cores.
    """

    # def __init__(
    #     self,
    #     estimators: list[tuple[str, object]],
    #     weights: 'Optional[ArrayLike]' = None,
    #     n_jobs: int = 1,
    # ):
    #     self.estimators = estimators
    #     self.weights = weights
    #     self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    def fit(self, X: 'ArrayLike', y: 'ArrayLike') -> 'VotingClassifier':
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        return super().fit(X, y)

    def predict(self, X: 'ArrayLike') -> 'NDArray':
        """
        Predict using the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        predictions = np.array([estimator.predict(X) for _, estimator in self.estimators])
        return np.apply_along_axis(
            lambda x: np.argmax(np.bincount(x, weights=self.weights)),
            axis=0,
            arr=predictions,
        )

    def _has_predict_proba(self) -> bool:
        """
        Check if the estimators have a predict_proba method.

        Returns
        -------
        has_proba : bool
            True if all estimators have a predict_proba method, False otherwise.
        """
        return all(hasattr(est, "predict_proba") for _, est in self.estimators)

    @available_if(_has_predict_proba)
    def predict_proba(self, X: 'ArrayLike') -> 'NDArray':
        """
        Predict the probabilities of the classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples, n_classes)
            The predicted probabilities of the classes.
        """
        proba_predictions = [est.predict_proba(X) for _, est in self.estimators]
        return np.average(proba_predictions, axis=0, weights=self.weights)

    def score(self, X: 'ArrayLike', y: 'ArrayLike') -> float:
        """
        Return the accuracy of the model on the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        score : float
            The accuracy score of the model on the input data.
        """
        return accuracy_score(y, self.predict(X))

class FairVotingClassifier(VotingClassifier):
    """
    Voting classifier that assigns a fairness-aware weight
    to each estimator before performing the vote.
    """
    def __init__(self,estimators,weights=None,alpha=0.5, n_jobs=1,):
        super().__init__(estimators=estimators,weights=weights,n_jobs=n_jobs,)
        self.alpha = alpha
        self.fair_weights = None
    def _compute_weight(self, accuracy, fairness):

        fairness_score = 1 - abs(fairness)
        return (self.alpha * accuracy + (1 - self.alpha) * fairness_score)

    def fit(self,X_train,y_train,X_val,y_val,A_val,):
        super().fit(X_train, y_train)
        weights = []
        for _, estimator in self.estimators:

            prediction = estimator.predict(X_val)

            acc = accuracy_score(
                y_val,
                prediction,
            )

            fairness = demographic_parity_difference(
                y_true=y_val,
                y_pred=prediction,
                sensitive_features=A_val,
            )

            weight = self._compute_weight(
                acc,
                fairness,
            )

            weights.append(weight)

        weights = np.array(weights)

        weights = weights / weights.sum()

        self.fair_weights = weights

        return self

    def predict(self, X):

        predictions = np.array(
            [
                estimator.predict(X)
                for _, estimator in self.estimators
            ]
        )

        return np.apply_along_axis(
            lambda x: np.argmax(
                np.bincount(
                    x.astype(int),
                    weights=self.fair_weights,
                )
            ),
            axis=0,
            arr=predictions,
        )
    
class VotingRegressor(BaseVoting):
    """
    A voting regressor that uses the average of the predictions.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        The list of estimators to fit and vote on.

    weights : array-like of shape (n_classifiers,) or None, default=None
        The weights of the classifiers. If None, then uniform weights are used.

    n_jobs : int, default=1
        The number of jobs to run in parallel.
        If -1, then the number of jobs is set to the number of CPU cores.
    """

    # def __init__(
    #     self,
    #     estimators: list[tuple[str, object]],
    #     weights: 'Optional[ArrayLike]' = None,
    #     n_jobs: int = 1,
    # ):
    #     self.estimators = estimators
    #     self.weights = weights
    #     self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

    def fit(self, X: 'ArrayLike', y: 'ArrayLike') -> 'VotingRegressor':
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        return super().fit(X, y)

    def predict(self, X: 'ArrayLike') -> 'NDArray':
        """
        Predict using the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array-like of shape (n_samples,)
            The predicted classes.
        """
        predictions = [est.predict(X) for _, est in self.estimators]
        return np.average(predictions, axis=0, weights=self.weights)

    def score(self, X: 'ArrayLike', y: 'ArrayLike') -> float:
        """
        Return the R2 score of the model on the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        score : float
            The R2 score of the model on the input data.
        """
        return r2_score(y, self.predict(X))



class FairVotingRegressor(VotingRegressor):
    """
    Voting regressor that assigns a fairness-aware weight
    to each estimator before performing the weighted average.
    """

    def __init__(
        self,
        estimators,
        weights=None,
        alpha=0.5,
        n_jobs=1,
    ):
        super().__init__(
            estimators=estimators,
            weights=weights,
            n_jobs=n_jobs,
        )

        self.alpha = alpha
        self.fair_weights = None

    def _compute_weight(self, performance, fairness_gap):
        # r2_score can be negative for a bad model; clip to 0 so it can't
        # produce a negative weight (accuracy_score never needed this).
        performance_score = max(performance, 0.0)

        # RMSE difference has no fixed upper bound (depends on target scale),
        # unlike demographic_parity_difference which is capped at 1.
        # Squash it into (0, 1]: gap=0 -> 1 (perfectly fair), gap large -> ~0.
        fairness_score = 1.0 / (1.0 + fairness_gap)

        return self.alpha * performance_score + (1 - self.alpha) * fairness_score

    def fit(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        A_val,
    ):
        super().fit(X_train, y_train)

        weights = []

        for _, estimator in self.estimators:
            prediction = estimator.predict(X_val)

            perf = r2_score(y_val, prediction)

            # Per-group RMSE, then the gap between the best- and worst-served group
            mf = MetricFrame(
                metrics=root_mean_squared_error,
                y_true=y_val,
                y_pred=prediction,
                sensitive_features=A_val,
            )
            fairness_gap = mf.difference()

            weight = self._compute_weight(perf, fairness_gap)
            weights.append(weight)

        weights = np.array(weights)
        total = weights.sum()
        self.fair_weights = weights / total if total > 0 else np.ones_like(weights) / len(weights)

        return self

    def predict(self, X):
        predictions = np.array(
            [estimator.predict(X) for _, estimator in self.estimators]
        )
        return np.average(predictions, axis=0, weights=self.fair_weights)
