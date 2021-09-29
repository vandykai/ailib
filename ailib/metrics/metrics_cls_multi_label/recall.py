"""Recall metric for Classification."""
import numpy as np
from ailib.metrics.base_metric import ClassificationMultiLabelMetric
from ailib.metrics.utils import sigmoid

class RecallMultiLabel(ClassificationMultiLabelMetric):
    """Recall metric."""

    ALIAS = ['recall']

    def __init__(self, threshold = 0.5):
        """:class:`Recall` constructor."""
        super().__init__()
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.recalls = []

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}@{self.threshold}"

    def __call__(self, y_true: list, y_pred: list) -> float:
        """
        Calculate recall.

        Example:
            >>> y_true = [[1, 0, 1, 0], [0, 1, 0, 1]]
            >>> y_pred = [[0, 0.4, 1.7, 0], [0, 0.5, 0.1, 0.6]]
            >>> AccuracyMultiLabel()(y_true, y_pred)
            0.75

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Recall.
        """
        recalls = self._compute(y_true, y_pred)
        return np.mean(recalls).item()

    def _compute(self, y_true: list, y_pred: list) -> list:
        y_true = np.array(y_true, dtype=np.int8)
        y_pred = np.array(y_pred, dtype=np.float64)
        y_pred = sigmoid(y_pred)
        y_pred = (y_pred > self.threshold).astype(np.int8)

        right = ((y_true + y_pred) >= 2).sum(axis=1)
        origin = y_true.sum(axis=1)
        recalls = np.divide(right, origin, out=np.zeros_like(right, dtype=np.float64), where=origin!=0)
        return recalls.tolist()

    def update(self, y_true: list, y_pred: list):
        recalls = self._compute(y_true, y_pred)
        self.recalls.extend(recalls)

    def result(self):
        return {'_score':np.mean(self.recalls).item()}