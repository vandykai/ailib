"""Metric base class and some related utilities."""

import abc

import numpy as np


class BaseMetric(abc.ABC):
    """Metric base class."""

    ALIAS = 'base_metric'

    def __init__(self):
        self.y_trues = []
        self.y_preds = []

    @abc.abstractmethod
    def __call__(self, y_true: list, y_pred: list) -> float:
        """
        Call to compute the metric.

        :param y_true: An list of groud truth labels.
        :param y_pred: An list of predicted values.
        :return: Evaluation of the metric.
        """

    def update(self, y_true: list, y_pred: list):
        self.y_trues.append(y_true)
        self.y_preds.append(y_pred)

    def result():
        y_true = [ex for item in self.y_trues for ex in item]
        y_pred = [ex for item in self.y_preds for ex in item]
        return self(y_true, y_pred)

    @abc.abstractmethod
    def __repr__(self):
        """:return: Formated string representation of the metric."""

    def __eq__(self, other):
        """:return: `True` if two metrics are equal, `False` otherwise."""
        return (type(self) is type(other)) and (vars(self) == vars(other))

    def __hash__(self):
        """:return: Hashing value using the metric as `str`."""
        return str(self).__hash__()


class RankingMetric(BaseMetric):
    """Ranking metric base class."""

    ALIAS = 'ranking_metric'


class ClassificationMetric(BaseMetric):
    """Rangking metric base class."""

    ALIAS = 'classification_metric'

class ClassificationMultiLabelMetric(BaseMetric):
    """Rangking metric base class."""

    ALIAS = 'classification_multi_label_metric'


def sort_and_couple(labels: np.array, scores: np.array) -> np.array:
    """Zip the `labels` with `scores` into a single list."""
    couple = list(zip(labels, scores))
    return np.array(sorted(couple, key=lambda x: x[1], reverse=True))
