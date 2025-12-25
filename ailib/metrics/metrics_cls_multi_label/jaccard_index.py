"""Jaccard metric for Classification."""
import numpy as np
from ailib.metrics.base_metric import ClassificationMultiLabelMetric
from ailib.metrics.utils import sigmoid

class JaccardMultiLabel(ClassificationMultiLabelMetric):
    """Jaccard metric."""

    ALIAS = ['jaccard', 'jaccard index']

    def __init__(self, threshold = 0.5):
        """:class:`Jaccard` constructor."""
        super().__init__()
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.jaccards = []

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}@{self.threshold}"

    def __call__(self, y_true: list, y_pred: list) -> float:
        """
        Calculate jaccard.

        Example:
            >>> y_true = [[1, 0, 1, 0], [0, 1, 0, 1]]
            >>> y_pred = [[0, 0.4, 1.7, 0], [0, 0.5, 0.1, 0.6]]
            >>> JaccardMultiLabel()(y_true, y_pred)
            0.5

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Jaccard.
        """
        jaccards = self._compute(y_true, y_pred)
        return np.mean(jaccards).item()

    def _compute(self, y_true: list, y_pred: list) -> list:
        y_true = np.array(y_true, dtype=np.int8)
        y_pred = np.array(y_pred, dtype=np.float64)
        y_pred = sigmoid(y_pred)
        y_pred = (y_pred > self.threshold).astype(np.int8)

        right = ((y_true + y_pred) >= 2).sum(axis=1)
        mix_origin = ((y_true + y_pred) >= 1).sum(axis=1)
        jaccards = np.divide(right, mix_origin, out=np.zeros_like(right, dtype=np.float64), where=mix_origin!=0)
        return jaccards.tolist()

    def update(self, y_true: list, y_pred: list):
        jaccards = self._compute(y_true, y_pred)
        self.jaccards.extend(jaccards)

    def result(self):
        return {'_score':np.mean(self.jaccards).item()}