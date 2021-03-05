import torch
import torch.nn as nn
import torch.nn.functional as F

class ELRMultiLabelLoss(nn.Module):
    def __init__(self, num_examp, n_classes, lambda_ = 3, beta=0.7, device=None):
        r"""Early Learning Regularization.
        Parameters
        * `num_examp` Total number of training examples.
        * `n_classes` Number of classes in the classification problem.
        * `lambda_` Regularization strength; must be a positive float, controling the strength of the ELR.
        * `beta` Temporal ensembling momentum for target estimation.
        """
        super().__init__()
        self.n_classes = n_classes
        self.target = torch.zeros(num_examp, self.n_classes, device=device)
        self.beta = beta
        self.lambda_ = lambda_

    def forward(self, index, output, label):
        r"""Early Learning Regularization.
        Args
        * `index` Training sample index, used to track training examples in different iterations.
        * `output` Model's prediction, same as PyTorch provided functions.
        * `label` Labels, same as PyTorch provided loss functions.
        """

        y_pred = output.sigmoid()
        y_pred = torch.clamp(y_pred, 1e-4, 1.0-1e-4)
        y_pred_ = y_pred.data.detach()
        self.target[index] = self.beta * self.target[index] + (1-self.beta) * (y_pred_/y_pred_.sum(dim=1,keepdim=True))
        ce_loss = F.binary_cross_entropy(output.sigmoid(), label)
        elr_reg = ((1-(self.target[index] * y_pred)).sum(dim=1).log()).mean()
        final_loss = ce_loss +  self.lambda_ *elr_reg
        return final_loss