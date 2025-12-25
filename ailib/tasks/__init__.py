from .classification import ClassificationTask
from .ranking import RankingTask
from .classification_multi_label import ClassificationMultiLabelTask
from .regression import RegressionTask
from .ner import NerTask

__all__ = [
    ClassificationTask,
    RankingTask,
    ClassificationMultiLabelTask,
    RegressionTask,
    NerTask
]