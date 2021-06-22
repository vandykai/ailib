import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, smoothing_eps, n_classes, ignore_index=-100, device=None):
        assert 0.0 <= smoothing_eps <= 1.0
        super().__init__()
        self.smoothing_value = smoothing_eps / (n_classes - 1)
        self.confidence = 1.0 - smoothing_eps
        self.ignore_index = ignore_index
        self.n_classes = n_classes

    def forward(self, pred, target):
        """
        pred (FloatTensor): [N, C, d1, d2, ..., dk]
        target (LongTensor): [N, d1, d2, ..., dk]
        """
        pred = F.log_softmax(pred, dim=1)
        model_prob = torch.full_like(pred, self.smoothing_value)
        model_prob.scatter_(1, target.unsqueeze(1).clamp(0, self.n_classes-1), self.confidence)
        loss = F.kl_div(pred, model_prob, reduction='none').sum(dim=1)
        ignore_mask = target.ne(self.ignore_index)
        loss = loss.masked_select(ignore_mask).mean()  # average later
        return loss

def single_label_loss(pred, target, trg_pad_idx=-1, smoothing=False):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''
    target = target.contiguous().view(-1)
    if smoothing:
        smoothing_eps = 0.1
        n_class = pred.size(1)
        one_hot = torch.zeros_like(pred).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - smoothing_eps) + (1 - one_hot) * smoothing_eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        non_pad_mask = target.ne(trg_pad_idx)
        loss = -(one_hot * log_prb).sum(dim=1)
        loss = loss.masked_select(non_pad_mask).mean()  # average later
    else:
        loss = F.cross_entropy(pred, target, ignore_index=trg_pad_idx, reduction='mean')
    return loss