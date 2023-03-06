import torch
import math
from scipy.special import comb
from torch._six import inf
from typing import Union, Iterable
import numpy as np
import random
import pandas as pd

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
 
def _confidence(click_num, impression_num, z):
    #z = 2.58 -> 99% confidence
    #z = 1.96 -> 95% confidence
    #z = 1.65 -> 90% confidence
    n = impression_num
    if n == 0: return 0
    phat = float(click_num) / n
    denorm = 1. + (z*z/n)
    enum1 = phat + z*z/(2*n)
    enum2 = z * math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))
    return (enum1-enum2)/denorm, (enum1+enum2)/denorm
 
def wilson_smooth(click_num, impression_num, z=1.96):
    r"""Caculate wilson smooth.
    Arguments:
        click_num: the number of clicked items
        impression_num: the number of exposure items
    Returns:
        wilson smooth value
    """
    if impression_num == 0:
        return 0
    else:
        return _confidence(click_num, impression_num, z)

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
    if len(parameters) == 0:
        return 0.
    with torch.no_grad():
        if norm_type == inf:
            total_norm = max(p.grad.abs().max() for p in parameters).tolist()
        else:
            total_norm = torch.norm(torch.stack([torch.norm(p.grad, norm_type) for p in parameters]), norm_type).tolist()
        return total_norm

def regularization(parameters: _tensor_or_tensors, norm_type: float = 2.0):
    r"""Caculate param norm of an iterable of parameters.
    
    The norm is computed over all tensor together, as if they were
    concatenated into a single vector.
    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    mean_norm = torch.mean(torch.stack([torch.norm(p, norm_type) for p in parameters]))
    return mean_norm

class BeyesianSmooth(object):
    r"""
    Examples:
        >>> beyesian_smooth = BeyesianSmooth()
        >>> clicks, impressions = beyesian_smooth.sample_from_beta(100, 1000, 10000, 1000)
        >>> beyesian_smooth.set_from_data_by_moment(clicks, impressions)
        >>> print(beyesian_smooth.alpha, beyesian_smooth.beta)
        >>> print(beyesian_smooth.beyesian_smooth(3000,4000))
        101.53840399337906 1014.9476473107375
        0.6061852554455499
    Reference:
        https://www.cnblogs.com/bentuwuying/p/6389222.html
        https://www.zhihu.com/question/30269898
    """
    def __init__(self):
        self.alpha = None
        self.beta = None

    def sample_from_beta(self, alpha: float, beta: float, sample_nums: int, impression_upperbound: int):
        sample = np.random.beta(alpha, beta, sample_nums)
        impressions = []
        clicks = []
        for click_ratio in sample:
            impression = random.random() * impression_upperbound
            click = impression * click_ratio
            impressions.append(impression)
            clicks.append(click)
        return clicks, impressions

    def set_from_data_by_moment(self, clicks: list, impressions: list):
        '''estimate alpha, beta using moment estimation'''
        mean, var = self.__compute_moment(clicks, impressions)
        if mean == 0:
            mean = 0.000001 
        if var == 0:
            var = 0.000001
        self.alpha = mean*(mean*(1-mean)/var-1)
        #self.alpha = (mean+0.000001) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)
        self.beta = (1-mean)*(mean*(1-mean)/var-1)
        #self.beta = (1.000001 - mean) * ((mean+0.000001) * (1.000001 - mean) / (var+0.000001) - 1)

    def beyesian_smooth(self, click_num: int, impression_num: int):
        return (self.alpha + click_num) / (self.alpha + self.beta + impression_num)

    def __compute_moment(self, clicks: list, impressions: list):
        '''moment estimation'''
        clicks = np.array(clicks)
        impressions = np.array(impressions)
        click_rates = clicks/impressions
        return click_rates.mean(), click_rates.var()

def distribution_ks(x1, x2):
    x1 = sorted(x1)
    x2 = sorted(x2)
    y1_fraction = []
    y2_fraction = []
    x1_idx, x2_idx = 0, 0
    x = []
    while x1_idx < len(x1) and x2_idx < len(x2):
        if x1[x1_idx] < x2[x2_idx]:
            x.append(x1[x1_idx])
            x1_idx += 1
        elif x1[x1_idx] > x2[x2_idx]:
            x.append(x2[x2_idx])
            x2_idx += 1
        else:
            x.append(x1[x1_idx])
            x1_idx += 1 
            x2_idx += 1
        y1_fraction.append(x1_idx/len(x1))
        y2_fraction.append(x2_idx/len(x2))
    while x1_idx < len(x1):
        x.append(x1[x1_idx])
        x1_idx += 1
        y1_fraction.append(x1_idx/len(x1))
        y2_fraction.append(x2_idx/len(x2))
    while x2_idx < len(x2):
        x.append(x2[x2_idx])
        x2_idx += 1
        y1_fraction.append(x1_idx/len(x1))
        y2_fraction.append(x2_idx/len(x2))
    return np.array(y1_fraction), np.array(y2_fraction), np.array(x)

def get_sample_rate_for_equal_dist(mark_dist, sample_dist, max_sample_rate=None):
    sample_rate = {}
    total_rate = sum(sample_dist.values())/sum(mark_dist.values())
    for key in sample_dist:
        sample_rate[key] = mark_dist[key]*total_rate/sample_dist[key] if key in mark_dist and sample_dist[key]!= 0 else 0
    if not max_sample_rate:
        max_sample_rate = max(sample_rate.values())
    for key in sample_rate:
        sample_rate[key] = sample_rate[key]/ max_sample_rate
    print(f"样本预估数:{sum(sample_dist.values())/max_sample_rate}")
    return sample_rate

def get_distribute(x, bins = 10, min_value = 0, max_value = 1):
    step = (max_value-min_value)/bins
    max_value = max_value + step
    dict_value = pd.cut(x, bins = np.arange(0, 1, step)).value_counts()
    dict_value = {str(k):v for k, v in dict_value.items()}
    return dict_value