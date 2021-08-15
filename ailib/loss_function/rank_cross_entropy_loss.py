"""The rank cross entropy loss."""
import torch
from torch import nn
import torch.nn.functional as F


class RankCrossEntropyLoss(nn.Module):
    """Creates a criterion that measures rank cross entropy loss."""

    __constants__ = ['num_neg']

    def __init__(self, num_neg: int = 1):
        """
        :class:`RankCrossEntropyLoss` constructor.

        :param num_neg: Number of negative instances in hinge loss.
        """
        super().__init__()
        self.num_neg = num_neg

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """
        Calculate rank cross entropy loss.

        :param y_pred: Predicted result. [N*(num_pos+num_neg), 1]
        :param y_true: Label. [N*(num_pos+num_neg)]
        :return: Rank cross loss.
        """

        # change [[pos], [neg], [neg], [neg],...,[pos], [neg], [neg], [neg],...]] to [[pos, neg, neg, neg,...],[pos, neg, neg, neg,...]]
        logits = y_pred.reshape(-1, self.num_neg + 1)
        labels = y_true.reshape(-1, self.num_neg + 1)
        return -torch.sum(labels * torch.log_softmax(logits, dim=-1), dim=-1).mean()
        # logits = y_pred[::(self.num_neg + 1), :]
        # labels = y_true[::(self.num_neg + 1), :]
        # for neg_idx in range(self.num_neg):
        #     neg_logits = y_pred[(neg_idx + 1)::(self.num_neg + 1), :]
        #     neg_labels = y_true[(neg_idx + 1)::(self.num_neg + 1), :]
        #     logits = torch.cat((logits, neg_logits), dim=-1)
        #     labels = torch.cat((labels, neg_labels), dim=-1)
        # return -torch.mean(
        #     torch.sum(
        #         labels * torch.log(F.softmax(logits, dim=-1) + torch.finfo(float).eps),
        #         dim=-1
        #     )
        # )

    @property
    def num_neg(self):
        """`num_neg` getter."""
        return self._num_neg

    @num_neg.setter
    def num_neg(self, value):
        """`num_neg` setter."""
        self._num_neg = value
