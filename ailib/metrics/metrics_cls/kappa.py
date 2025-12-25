"""Kappa metric for Classification."""
import numpy as np
from ailib.metrics.base_metric import ClassificationMetric
from sklearn.metrics import cohen_kappa_score

class Kappa(ClassificationMetric):
    """Kappa metric."""

    ALIAS = ['kappa']

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def __call__(self, y_true: list, y_pred: list) -> float:
        """
        Calculate kappa.

        Example:
            >>> y_true = [1 ,2, 1, 1, 1, 0]
            >>> y_pred = [[0.1, 0.7, 0.2], [0.1, 0.7, 0.2], [0.1, 0.7, 0.2], [0.1, 0.7, 0.2], [0.1, 0.7, 0.2], [0.1, 0.7, 0.2]]
            >>> Kappa()(y_true, y_pred)
            0.0

        :param y_true: The ground true label of each example.
        :param y_pred: The predicted scores of each example.
        :return: Kappa.
        """
        y_true = np.array(y_true, dtype=np.int8)
        y_pred = np.array(y_pred, dtype=np.float64)
        y_pred = np.argmax(y_pred, axis=-1)
        return cohen_kappa_score(y_true, y_pred)