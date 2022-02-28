from .cls_early_learning import ELRLoss
from .cls_label_smooth import LabelSmoothingLoss
from .cls_multi_label_early_learning import ELRMultiLabelLoss
from .cls_multi_label_softmax import SoftmaxMultiLabelLoss
from .rank_cross_entropy_loss import RankCrossEntropyLoss
from .rank_hinge_loss import RankHingeLoss

__all__ = [
    'ELRLoss',
    'LabelSmoothingLoss',
    'ELRMultiLabelLoss',
    'SoftmaxMultiLabelLoss',
    'RankCrossEntropyLoss',
    'RankHingeLoss'
]