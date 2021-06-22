"""An implementation of DSSM, Deep Structured Semantic Model."""
import typing

import torch
import torch.nn.functional as F

from matchzoo import preprocessors
from matchzoo.engine.param_table import ParamTable
from matchzoo.engine.param import Param
from matchzoo.engine.base_model import BaseModel
from matchzoo.engine.base_preprocessor import BasePreprocessor

from ailib.models.base_model import BaseModel
from ailib.models.base_model_param import BaseModelParam
from ailib.modules.attention import Attention
from ailib.modules.matching import Matching
from ailib.param.param import Param
from ailib.param import hyper_spaces
from ailib.param.param_table import ParamTable
from ailib.tools.utils_name_parse import parse_activation

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
        self.mlp_left = self._make_multi_layer_perceptron_layer(
            self.config.vocab_size
        )
        self.mlp_right = self._make_multi_layer_perceptron_layer(
            self.config.vocab_size
        )
        self.out = self._make_output_layer(1)

    def forward(self, inputs):
        """Forward."""
        # Process left & right input.
        input_left, input_right = inputs['ngram_left'], inputs['ngram_right']
        input_left = self.mlp_left(input_left)
        input_right = self.mlp_right(input_right)

        # Dot product with cosine similarity.
        x = F.cosine_similarity(input_left, input_right)

        out = self.out(x.unsqueeze(dim=1))
        return out
