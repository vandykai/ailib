import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftmaxMultiLabelLoss(nn.Module):
    def __init__(self):
        r"""
        ref: https://kexue.fm/archives/7359
        Parameters
        """
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        r"""Early Learning Regularization.
        Args
        * `y_pred` [[0.1, 0.9, 0.7], [0.8, 0.1, 0.1]].
        * `y_true` [[0, 0, 1], [1, 0, 0]].
        Example:
            >>> import torch
            >>> from ailib.loss_function.cls_multi_label_softmax import SoftmaxMultiLabelLoss
            >>> loss_func = SoftmaxMultiLabelLoss()
            >>> y_pred = torch.tensor([[0.1, 0.8, 0.9], [0.8, 0.2, 0.1]])
            >>> y_true = torch.tensor([[0, 1, 1], [1, 0, 0]])
            >>> loss(a, b)
            tensor([1.3628, 1.5730])
        """
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], axis=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], axis=-1)
        neg_loss = torch.logsumexp(y_pred_neg, axis=-1)
        pos_loss = torch.logsumexp(y_pred_pos, axis=-1)
        total_loss = neg_loss + pos_loss
        total_loss = total_loss.mean()
        return total_loss