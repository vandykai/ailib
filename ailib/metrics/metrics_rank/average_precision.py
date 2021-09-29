"""Average precision metric for ranking."""
import numpy as np
from ailib.metrics.base_metric import RankingMetric
from ailib.metrics.metrics_rank.precision import Precision

class AveragePrecision(RankingMetric):
    """Average precision metric."""

    ALIAS = ['average_precision', 'ap']

    def __init__(self, threshold: float = 0.):
        """
        :class:`AveragePrecision` constructor.

        :param threshold: The label threshold of relevance degree.
        """
        self._threshold = threshold
        self.reset()

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}({self._threshold})"

    def reset(self):
        self.aps = []

    def _compute(self, y_true: list, y_pred: list) -> float:
        precision_metrics = [Precision(k + 1) for k in range(len(y_pred))]
        out = [metric(y_true, y_pred) for metric in precision_metrics]
        if not out:
            return 0.
        return np.mean(out).item()

    def update(self, y_true: list, y_pred: list):
        ap = self._compute(y_true, y_pred)
        self.aps.append(ap)

    def __call__(self, y_true: list, y_pred: list) -> float:
        """
        Calculate average precision (area under PR curve).

        Example:
            >>> y_true = [0, 1]
            >>> y_pred = [0.1, 0.6]
            >>> round(AveragePrecision()(y_true, y_pred), 2)
            0.75
            >>> round(AveragePrecision()([], []), 2)
            0.0

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Average precision.
        """

    def result(self):
        return {'_score':np.mean(self.aps).item()}
