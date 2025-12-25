"""Relative discounted cumulative gain metric for ranking."""
import numpy as np
from ailib.metrics.metrics_rank.discounted_cumulative_gain import DiscountedCumulativeGain
from ailib.metrics.base_metric import RankingMetric

class RelativeDiscountedCumulativeGain(RankingMetric):
    """Relative discounted cumulative gain metric."""

    ALIAS = ['relative_discounted_cumulative_gain', 'ndcg']

    def __init__(self, k: int = -1, threshold: float = 0.):
        """
        :class:`RelativeDiscountedCumulativeGain` constructor.

        :param k: Number of results to consider
        :param threshold: the label threshold of relevance degree.
        """
        self._k = k
        self._threshold = threshold
        self.reset()

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}@{self._k}({self._threshold})"

    def reset(self):
        self.ndcgs = []

    def _compute(self, y_true: list, y_pred: list) -> float:
        """
        Calculate accuracy.

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Accuracy list.
        """
        dcg_metric = DiscountedCumulativeGain(k=self._k, threshold=self._threshold)
        idcg_val = dcg_metric(y_true, y_true)
        y_true_reversed = [-1 * it for it in y_true]
        bdcg_val = dcg_metric(y_true, y_true_reversed)
        dcg_val = dcg_metric(y_true, y_pred)
        return (dcg_val-bdcg_val)/ (idcg_val-bdcg_val) if idcg_val != 0 else 0.

    def update(self, y_true: list, y_pred: list):
        ndcg = self._compute(y_true, y_pred)
        self.ndcgs.append(ndcg)

    def __call__(self, y_true: list, y_pred: list) -> float:
        """
        Calculate relative discounted cumulative gain (ndcg).

        Relevance is positive real values or binary values.

        Example:
            >>> y_true = [0, 1, 2, 0]
            >>> y_pred = [0.4, 0.2, 0.5, 0.7]
            >>> rdcg = RelativeDiscountedCumulativeGain
            >>> rdcg(k=1)(y_true, y_pred)
            0.0
            >>> round(rdcg(k=2)(y_true, y_pred), 2)
            0.52
            >>> round(rdcg(k=3)(y_true, y_pred), 2)
            0.44
            >>> type(rdcg()(y_true, y_pred))
            <class 'float'>

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.

        :return: Relative discounted cumulative gain.
        """
        return self._compute(y_true, y_pred)

    def result(self):
        return {'_score':np.mean(self.ndcgs).item()}