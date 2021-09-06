"""Accuracy metric for Classification."""
import scipy
from ailib.metrics.base_metric import RegressionMetric

class CorrCoef(RegressionMetric):
    """Regression metric."""

    ALIAS = ['corr']

    def __init__(self, method = 'pearson'):
        """:class:`CORR` constructor."""
        super().__init__()
        if method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError(
                "method must be either 'pearson', "
                "'spearman', 'kendall', or a callable, "
                f"'{method}' was supplied"
            )
        self._method = method

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
            >>> CorrCoef()(y_true, y_pred)
            1.0

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: CORR.
        """
        if self._method == 'pearson':
            correlation, pvalue = scipy.stats.pearsonr(y_true, y_pred)
        elif self._method == 'spearman':
            correlation, pvalue = scipy.stats.spearmanr(y_true, y_pred)
        elif self._method == 'kendall':
            correlation, pvalue = scipy.stats.kendalltau(y_true, y_pred)
        return correlation