"""KS metric for Classification."""
import numpy as np
from ailib.metrics.base_metric import ClassificationMetric
from sklearn.metrics import roc_curve, auc

class KS(ClassificationMetric):
    """KS metric."""

    ALIAS = ['ks']

    def __init__(self, pos_label=1):
        super().__init__()
        self.pos_label = pos_label

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def __call__(self, y_true: list, y_pred: list) -> float:
        """
        Calculate KS.

        Example:
            >>> y_true = [1 ,0, 1, 1, 1, 0]
            >>> y_pred = [[0.3, 0.7], [0.5, 0.5], [0.4, 0.6], [0.6, 0.4], [0.62, 0.38], [0.35, 0.65]]
            >>> KS()(y_true, y_pred)
            0.25

        :param y_true: The ground true label of each example.
        :param y_pred: The predicted scores of each example.
        :return: KS.
        """
        y_true = np.array(y_true, dtype=np.int8)
        y_pred = np.array(y_pred, dtype=np.float64)
        y_pred = y_pred[:, self.pos_label]
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=self.pos_label, drop_intermediate=False)
        return max(tpr-fpr)