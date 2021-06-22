import collections
import typing

import numpy as np

from .unit import Unit


class BatchPadding2D(Unit):
    """
    Batch Padding unit.

    :param pad_token: Padding token(anything like id or str).
    :param mode: One of `pre`(padding first), `post`(padding last).
    :param fix_length: int type.
    :param include_lengths: Returns a tuple of the padded list and a list containing lengths 
        of each example if `include_lengths` is `True`, else just returns the padded list
    :param init_token: init token added to the head of each sentence.
    :param eos_token: eos token added to the end of each sentence.
    :param truncate_model: When fix_length is setted, how to truncate each sentence, 
        One of `pre`(keep tail part of the sentence), `mid`(keep both ends of the sentence), `post`(keep head part of the sentence).

    Examples::
        >>> from ailib.preprocessors import units
        >>> batch_padding_2D = units.BatchPadding2D(
        ...     mode='pre', include_lengths=True)
        >>> batch_padding_2D.transform([['A', 'B', 'B'], ['C', 'C']])
        ([['A', 'B', 'B'], ['<pad>', 'C', 'C']], [3, 2])

    """

    def __init__(self, pad_token = '<pad>', mode: str = 'post', fix_length = None, 
                 include_lengths = False, init_token = None, eos_token = None,
                 truncate_model = 'pre'
                ):
        """Batch Padding unit."""
        super().__init__()
        self._pad_token = pad_token
        self._mode = mode
        self._fix_length = fix_length
        self._init_token = init_token
        self._eos_token = eos_token
        self._truncate_model = truncate_model
        self._include_lengths = include_lengths

    def transform(self, minibatch: typing.List[list]) -> typing.List[list]:
        """Pad a batch of examples.

        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True`, else just returns the padded list.
        """
        minibatch = list(minibatch)
        if self._fix_length is None:
            max_len = max(len(x) for x in minibatch)
        else:
            max_len = self._fix_length + (
                self._init_token, self._eos_token).count(None) - 2
        padded, lengths = [], []
        for x in minibatch:
            pad_part = [self._pad_token] * max(0, max_len - len(x))
            init_token_part = [] if self._init_token is None else [self._init_token]
            eos_token_part = [] if self._eos_token is None else [self._eos_token]
            if self._truncate_model == 'pre':
                token_part = list(x[-max_len:])
            elif self._truncate_model == 'post':
                token_part = list(x[:max_len])
            elif self._truncate_model == 'mid':
                token_part = list(x[:max_len//2]) + list(x[-max_len//2-1:])

            if self._mode == 'pre':
                padded.append(pad_part + init_token_part + token_part + eos_token_part)
            elif self._mode == 'post':
                padded.append(init_token_part + token_part + eos_token_part + pad_part)
            else:
                raise ValueError('{} is not a vaild pad mode.'.format(mode))
            lengths.append(len(padded[-1]) - max(0, max_len - len(x)))
        if self._include_lengths:
            return (padded, lengths)
        return padded

class BatchPadding3D(Unit):
    """
    Batch Padding unit.

    :param pad_token: Padding token(anything like id or str).
    :param mode: One of `pre`(padding first), `post`(padding last).
    :param fix_length: int type.
    :param include_lengths: Returns a tuple of the padded list and a list containing lengths 
        of each example if `include_lengths` is `True`, else just returns the padded list
    :param init_token: init token added to the head of each sentence.
    :param eos_token: eos token added to the end of each sentence.
    :param truncate_model: When fix_length is setted, how to truncate each sentence, 
        One of `pre`(keep tail part of the sentence), `mid`(keep both ends of the sentence), `post`(keep head part of the sentence).

    Examples::
        >>> from ailib.preprocessors import units
        >>> batch_padding_3D = units.BatchPadding3D(
        ...     mode='pre', include_lengths=True)
        >>> minibatch = [
        ...     [list('john'), list('loves'), list('mary')],
        ...     [list('mary'), list('cries')],
        ... ]
        >>> batch_padding_3D.transform(minibatch)
                ([[['<pad>', 'j', 'o', 'h', 'n'],
                    ['l', 'o', 'v', 'e', 's'],
                    ['<pad>', 'm', 'a', 'r', 'y']],
                    [['<pad>', '<pad>', '<pad>', '<pad>', '<pad>'],
                    ['<pad>', 'm', 'a', 'r', 'y'],
                    ['c', 'r', 'i', 'e', 's']]],
                    [3, 2],
                    [[4, 5, 4], [0, 4, 5]])
    """

    def __init__(self, pad_token = '<pad>', mode: str = 'post', fix_length = None, 
                 include_lengths = False, init_token = None, eos_token = None,
                 truncate_model = 'pre', nesting_pad = None
                ):
        """Batch Padding unit."""
        super().__init__()
        self._pad_token = pad_token
        self._mode = mode
        self._fix_length = fix_length
        self._init_token = init_token
        self._eos_token = eos_token
        self._truncate_model = truncate_model
        self._include_lengths = include_lengths
        self._nesting_pad = nesting_pad
        if self._nesting_pad is None:
            self._nesting_pad = BatchPadding2D(pad_token=pad_token, mode=mode, fix_length=fix_length, 
                                    include_lengths=True, init_token=init_token, eos_token=eos_token, truncate_model=truncate_model)
        else:
            self._nesting_pad._pad_token = pad_token

    def transform(self, minibatch: typing.List[list]) -> typing.List[list]:
        """Pad a batch of examples.

        Pads to self.fix_length if provided, otherwise pads to the length of
        the longest example in the batch. Prepends self.init_token and appends
        self.eos_token if those attributes are not None. Returns a tuple of the
        padded list and a list containing lengths of each example if
        `self.include_lengths` is `True`, else just returns the padded list.
        """
        minibatch = list(minibatch)
        if self._nesting_pad._fix_length is None:
            max_len = max(len(xs) for ex in minibatch for xs in ex)
            fix_len = max_len + 2 - (self._nesting_pad._init_token,
                                    self._nesting_pad._eos_token).count(None)
            self._nesting_pad._fix_length = fix_len
            self._nesting_pad._include_lengths = True
        pad_token = [self._pad_token] * self._nesting_pad._fix_length
        init_token = [self._init_token] if self._init_token is not None else None
        eos_token = [self._eos_token] if self._eos_token is not None else None
        batch_padding_2D = BatchPadding2D(pad_token=pad_token, mode=self._mode, fix_length=self._fix_length, 
                                include_lengths = True, init_token=init_token, eos_token=eos_token, truncate_model=self._truncate_model)
        padded, sentence_lengths = batch_padding_2D.transform(minibatch)
        padded_with_lengths = [self._nesting_pad.transform(ex) for ex in padded]
        word_lengths = []
        final_padded = []
        max_sen_len = len(padded[0])
        for (pad, lens), sentence_len in zip(padded_with_lengths, sentence_lengths):
            if sentence_len == max_sen_len:
                lens = lens
                pad = pad
            elif self._mode == 'pre':
                lens[:(max_sen_len - sentence_len)] = (
                    [0] * (max_sen_len - sentence_len))
                pad[:(max_sen_len - sentence_len)] = (
                   [pad_token] * (max_sen_len - sentence_len))
            elif self._mode == 'post':
                lens[-(max_sen_len - sentence_len):] = (
                    [0] * (max_sen_len - sentence_len))
                pad[-(max_sen_len - sentence_len):] = (
                   [pad_token] * (max_sen_len - sentence_len))
            else:
                raise ValueError('{} is not a vaild pad mode.'.format(mode))
            word_lengths.append(lens)
            final_padded.append(pad)
        padded = final_padded
        if self._include_lengths:
            return padded, sentence_lengths, word_lengths
        return padded