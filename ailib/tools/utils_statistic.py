import torch
import math
from scipy.special import comb
from torch._six import inf
from typing import Union, Iterable
_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def comb_probability(p, click_list):
    k = sum(click_list)
    n = len(click_list)
    # math.pow(p, k)*math.pow(1-p, 1-k)* comb(n, k, exact=True)
    return math.exp(math.log(p)*k+math.log(1-p)*(n-k) + math.log(comb(n, k, exact=True)))
 
def _confidence(clicks, impressions, z):
    #z = 2.58 -> 99% confidence
    #z = 1.96 -> 95% confidence
    #z = 1.65 -> 90% confidence
    n = impressions
    if n == 0: return 0
    phat = float(clicks) / n
    denorm = 1. + (z*z/n)
    enum1 = phat + z*z/(2*n)
    enum2 = z * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))
    return (enum1-enum2)/denorm, (enum1+enum2)/denorm
 
def wilson_smooth(clicks, impressions, z=1.96):
    if impressions == 0:
        return 0
    else:
        return _confidence(clicks, impressions, z)

def grad_norm(parameters: _tensor_or_tensors, norm_type: float = 2.0) -> torch.Tensor:
    r"""Caculate gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return 0.
    with torch.no_grad():
        if norm_type == inf:
            total_norm = max(p.grad.abs().max() for p in parameters).tolist()
        else:
            total_norm = torch.norm(torch.stack([torch.norm(p.grad, norm_type) for p in parameters]), norm_type).tolist()
        return total_norm