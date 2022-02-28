# coding: UTF-8
import random
from typing import Tuple
import torch.nn as nn
import numpy as np
import torch
import os
from contextlib import contextmanager
from copy import deepcopy
from sklearn.model_selection import train_test_split

def seed_everything(seed=1029):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # may slow down the speed of convolution

class RandomShuffler(object):
    """Use random functions while keeping track of the random state to make it
    reproducible and deterministic."""

    def __init__(self, random_state=None):
        self._random_state = random_state
        if self._random_state is None:
            self._random_state = random.getstate()

    @contextmanager
    def use_internal_state(self):
        """Use a specific RNG state."""
        old_state = random.getstate()
        random.setstate(self._random_state)
        yield
        self._random_state = random.getstate()
        random.setstate(old_state)

    @property
    def random_state(self):
        return deepcopy(self._random_state)

    @random_state.setter
    def random_state(self, s):
        self._random_state = s

    def __call__(self, data):
        """Shuffle and return a new list."""
        with self.use_internal_state():
            return random.sample(data, len(data))

def rationed_split(examples, train_ratio, test_ratio, val_ratio, rnd=None):
    """Create a random permutation of examples, then split them by ratios

    Arguments:
        examples: a list of data
        train_ratio, test_ratio, val_ratio: split fractions.
        rnd: a random shuffler

    Examples:
        >>> examples = []
        >>> train_ratio, test_ratio, val_ratio = 0.7, 0.2, 0.1
        >>> rnd = torchtext.data.dataset.RandomShuffler(None)
        >>> train_examples, test_examples, valid_examples = \
                rationed_split(examples, train_ratio, test_ratio, val_ratio, rnd)
    """
    N = len(examples)
    if not rnd:
        rnd = RandomShuffler(None)
    randperm = rnd(range(N))
    train_len = int(round(train_ratio * N))

    # Due to possible rounding problems
    if not val_ratio:
        test_len = N - train_len
    else:
        test_len = int(round(test_ratio * N))

    indices = (randperm[:train_len],  # Train
               randperm[train_len:train_len + test_len],  # Test
               randperm[train_len + test_len:])  # Validation

    # There's a possibly empty list for the validation set
    data = tuple([examples[i] for i in index] for index in indices)

    return data

def rationed_split_df(df_examples, train_ratio, test_ratio, val_ratio, rnd=None):
    """Create a random permutation of examples, then split them by ratios

    Arguments:
        examples: a list of data
        train_ratio, test_ratio, val_ratio: split fractions.
        rnd: a random shuffler

    Examples:
        >>> examples = []
        >>> train_ratio, test_ratio, val_ratio = 0.7, 0.2, 0.1
        >>> rnd = torchtext.data.dataset.RandomShuffler(None)
        >>> train_examples, test_examples, valid_examples = \
                rationed_split(examples, train_ratio, test_ratio, val_ratio, rnd)
    """
    N = len(df_examples)
    if not rnd:
        rnd = RandomShuffler(None)
    randperm = rnd(range(N))
    train_len = int(round(train_ratio * N))

    # Due to possible rounding problems
    if not val_ratio:
        test_len = N - train_len
    else:
        test_len = int(round(test_ratio * N))

    indices = (randperm[:train_len],  # Train
               randperm[train_len:train_len + test_len],  # Test
               randperm[train_len + test_len:])  # Validation

    # There's a possibly empty list for the validation set
    data = tuple(df_examples.iloc[index].reset_index(drop=True) for index in indices)

    return data