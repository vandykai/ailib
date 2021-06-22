"""Accuracy metric for Classification."""
import numpy as np
from ailib.metrics.base_metric import ClassificationMetric

class Accuracy(ClassificationMetric):
    """Accuracy metric."""

    ALIAS = ['accuracy', 'acc']

    def __init__(self):
        """:class:`Accuracy` constructor."""
        super().__init__()

    def reset(self):
        self.accuracys = []

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def __call__(self, y_true: list, y_pred: list) -> float:
        """
        Calculate accuracy.

        Example:
            >>> import numpy as np
            >>> y_true = [1]
            >>> y_pred = [[0, 1]]
            >>> Accuracy()(y_true, y_pred)
            1.0

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Accuracy.
        """
        accuracys = self._compute(y_true, y_pred)
        return np.mean(accuracys)

    def _compute(self, y_true: list, y_pred: list) -> list:
        """
        Calculate accuracy.

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Accuracy list.
        """
        y_true = np.array(y_true, dtype=np.int8)
        y_pred = np.array(y_pred, dtype=np.float64)
        y_pred = np.argmax(y_pred, axis=-1)
        return y_pred == y_true

    def update(self, y_true: list, y_pred: list):
        accuracys = self._compute(y_true, y_pred)
        self.accuracys.extend(accuracys)

    def result(self):
        return np.mean(self.accuracys).item()