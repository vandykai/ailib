"""CrossEntropy metric for Classification."""
import numpy as np
from ailib.metrics.base_metric import ClassificationMultiLabelMetric
from ailib.metrics.utils import sigmoid

class CrossEntropy(ClassificationMultiLabelMetric):
    """Cross entropy metric."""

    ALIAS = ['cross_entropy', 'ce']

    def __init__(self):
        """:class:`CrossEntropy` constructor."""

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def __call__(
        self,
        y_true: list,
        y_pred: list,
        eps: float = 1e-12
    ) -> float:
        """
        Calculate cross entropy.

        Example:
            >>> y_true = [[0, 0, 1, 0], [0, 1, 0, 1]]
            >>> y_pred = [[0, 0.4, 1.7, 0], [0, 0.5, 0.1, 0.6]]
            >>> CrossEntropy()(y_true, y_pred)
            0.886249069137089

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :param eps: The Log loss is undefined for p=0 or p=1,
            so probabilities are clipped to max(eps, min(1 - eps, p)).
        :return: Average precision.
        """
        y_true = np.array(y_true, dtype=np.int8)
        y_pred = np.array(y_pred, dtype=np.float64)
        y_pred = sigmoid(y_pred)
        y_pred = np.clip(y_pred, eps, 1. - eps)

        return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_pred.shape[0]