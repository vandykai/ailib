"""An implementation of DSSM, Deep Structured Semantic Model."""
import typing

import torch
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

class Model(BaseModel):
    """
    Deep structured semantic model.

    Examples:
        >>> model_param = ModelParam()
        >>> model_param['mlp_num_layers'] = 3
        >>> model_param['mlp_num_units'] = 300
        >>> model_param['mlp_num_fan_out'] = 128
        >>> model_param['mlp_activation_func'] = 'tanh'
        >>> model = DSSM(model_param.to_config())

    """
    def __init__(self, config):
        """
        DSSM use Siamese arthitecture.
        """
        super().__init__()
        self.config = config
        self.left_embedding = self._make_default_embedding_layer()
        self.right_embedding = self._make_default_embedding_layer()
        self.multihead_attention = MultiheadAttention(self.config.embedding_output_dim, num_heads=1)
        self.mlp_left = self._make_multi_layer_perceptron_layer(
            self.config.embedding_output_dim
        )
        self.mlp_right = self._make_multi_layer_perceptron_layer(
            self.config.embedding_output_dim
        )
        self.out = self._make_output_layer(1)

    def forward(self, inputs):
        """Forward."""
        # Process left & right input
        input_left, input_right = inputs['ngram_left'], inputs['ngram_right']
        input_left_length, input_right_length = inputs['ngram_left_length'], inputs['ngram_right_length']
        left_key_padding_mask = torch.ones((input_left.size(0), input_left.size(1)), device=input_left.device).bool()
        for i in range(len(input_left_length)):
            left_key_padding_mask[i][:input_left_length[i]] = False
        right_key_padding_mask = torch.ones((input_right.size(0), input_right.size(1)), device=input_right.device).bool()
        for i in range(len(input_right_length)):
            right_key_padding_mask[i][:input_right_length[i]] = False
        input_left = self.left_embedding(input_left).transpose(0, 1)
        input_right = self.right_embedding(input_right).transpose(0, 1)
        input_left, _ = self.multihead_attention(input_left[:1], input_left, input_left, key_padding_mask=left_key_padding_mask)
        input_right, _ = self.multihead_attention(input_right[:1], input_right, input_right, key_padding_mask=right_key_padding_mask)

        input_left = self.mlp_left(input_left.squeeze(0))
        input_right = self.mlp_right(input_right.squeeze(0))

        # Dot product with cosine similarity.
        x = F.cosine_similarity(input_left, input_right)
        out = self.out(x.unsqueeze(dim=1))
        out = out.squeeze(-1)
        return out
