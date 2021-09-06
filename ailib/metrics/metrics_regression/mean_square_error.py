"""Accuracy metric for Classification."""
import numpy as np
from ailib.metrics.base_metric import RegressionMetric

class MSE(RegressionMetric):
    """Regression metric."""

    ALIAS = ['mse']

    def __init__(self):
        """:class:`MSE` constructor."""
        super().__init__()

    def reset(self):
        self.mses = []

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def __call__(self, y_true: list, y_pred: list) -> float:
        """
        Calculate mse.

        Example:
            >>> import numpy as np
            >>> y_true = [1.0, 2.0, 3.0]
            >>> y_pred = [2.0, 3.0, 4.0]
            >>> MSE()(y_true, y_pred)
            1.0

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: MSE.
        """
        mses = self._compute(y_true, y_pred)
        return np.mean(mses)

    def _compute(self, y_true: list, y_pred: list) -> list:
        """
        Calculate accuracy.

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: MSE list.
        """
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        return np.square(y_true - y_pred).tolist()

    def update(self, y_true: list, y_pred: list):
        mses = self._compute(y_true, y_pred)
        self.mses.extend(mses)

    def result(self):
        return np.mean(self.mses).item()