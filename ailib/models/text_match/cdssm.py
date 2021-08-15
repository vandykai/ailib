"""An implementation of DSSM, Deep Structured Semantic Model."""
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F

from ailib.models.base_model import BaseModel
from ailib.models.base_model_param import BaseModelParam
from ailib.modules.attention import Attention
from ailib.modules.matching import Matching
from ailib.param.param import Param
from ailib.param import hyper_spaces
from ailib.param.param_table import ParamTable
from ailib.tools.utils_name_parse import parse_activation
from ailib.modules.multihead_attention import MultiheadAttention

class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=True, with_multi_layer_perceptron=True):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self['model_name'] = "DSSM"
        self.add(Param(name='vocab_size', value=419,
                         desc="Size of vocabulary."))
        self.add(Param(name='filters', value=3,
                         desc="Number of filters in the 1D convolution layer."))
        self.add(Param(name='kernel_size', value=3,
                         desc="Number of kernel size in the 1D convolution layer."))
        self.add(Param(name='conv_activation_func', value='relu',
                         desc="Activation function in the convolution layer."))
        self.add(Param(name='dropout_rate', value=0.3, desc="The dropout rate."))

class Model(BaseModel):
    """
    CDSSM Model implementation.

    Learning Semantic Representations Using Convolutional Neural Networks
    for Web Search. (2014a)
    A Latent Semantic Model with Convolutional-Pooling Structure for
    Information Retrieval. (2014b)

    Examples:
        >>> model_param = ModelParam()
        >>> model_param['task'] = tasks.Ranking()
        >>> model_param['vocab_size'] = 4
        >>> model_param['filters'] = 32
        >>> model_param['kernel_size'] = 3
        >>> model_param['conv_activation_func'] = "relu"
        >>> model = CDSSM(model_param.to_config())

    """
    def __init__(self, config):
        """
        DSSM use Siamese arthitecture.
        """
        super().__init__()
        self.config = config
        self.net_left = self._create_base_network()
        self.net_right = self._create_base_network()
        self.out = self._make_output_layer(1)


    def _create_base_network(self) -> nn.Module:
        """
        Apply conv and maxpooling operation towards to each letter-ngram.

        The input shape is `fixed_text_length`*`number of letter-ngram`,
        as described in the paper, `n` is 3, `number of letter-trigram`
        is about 30,000 according to their observation.

        :return: A :class:`nn.Module` of CDSSM network, tensor in tensor out.
        """
        pad = nn.ConstantPad1d((0, self.config.kernel_size - 1), 0)
        conv = nn.Conv1d(
            in_channels=self.config.vocab_size,
            out_channels=self.config.filters,
            kernel_size=self.config.kernel_size
        )
        activation = parse_activation(
            self.config.conv_activation_func
        )
        dropout = nn.Dropout(p=self.config.dropout_rate)
        pool = nn.AdaptiveMaxPool1d(1)
        squeeze = Squeeze()
        mlp = self._make_multi_layer_perceptron_layer(
            self.config.filters
        )
        return nn.Sequential(
            pad, conv, activation, dropout, pool, squeeze, mlp
        )

    def forward(self, inputs):
        """Forward."""
        input_left, input_right = inputs['ngram_left'], inputs['ngram_right']
        input_left = input_left.transpose(1, 2)
        input_right = input_right.transpose(1, 2)
        input_left = self.net_left(input_left)
        input_right = self.net_right(input_right)

        # Dot product with cosine similarity.
        x = F.cosine_similarity(input_left, input_right)

        out = self.out(x.unsqueeze(dim=1))
        return out

class Squeeze(nn.Module):
    """Squeeze."""

    def forward(self, x):
        """Forward."""
        return x.squeeze(dim=-1)