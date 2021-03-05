"""Precision metric for Classification."""
import numpy as np
from ailib.metrics.base_metric import ClassificationMultiLabelMetric
from ailib.metrics.utils import sigmoid

class PrecisionMultiLabel(ClassificationMultiLabelMetric):
    """Precision metric."""

    ALIAS = ['precision']

    def __init__(self, threshold = 0.5):
        """:class:`Precision` constructor."""
        super().__init__()
        self.threshold = threshold
        self.precisions = []

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}@{self.threshold}"

    def __call__(self, y_true: list, y_pred: list) -> float:
        """
        Calculate precision.

        Example:
            >>> y_true = [[1, 0, 1, 0], [0, 1, 0, 1]]
            >>> y_pred = [[0, 0.4, 1.7, 0], [0, 0.5, 0.1, 0.6]]
            >>> PrecisionMultiLabel()(y_true, y_pred)
            0.5833333333333333

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Precision.
        """
        precisions = self._compute(y_true, y_pred)
        return np.mean(precisions).item()

    def _compute(self, y_true: list, y_pred: list) -> list:
        y_true = np.array(y_true, dtype=np.int8)
        y_pred = np.array(y_pred, dtype=np.float64)
        y_pred = sigmoid(y_pred)
        y_pred = (y_pred > self.threshold).astype(np.int8)

        right = ((y_true + y_pred) >= 2).sum(axis=1)
        found = y_pred.sum(axis=1)
        precisions = np.divide(right, found, out=np.zeros_like(right, dtype=np.float64), where=found!=0)
        return precisions.tolist()

    def update(self, y_true: list, y_pred: list):
        precisions = self._compute(y_true, y_pred)
        self.precisions.extend(precisions)

    def result(self):
        return np.mean(self.precisions).item()