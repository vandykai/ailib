"""An implementation of DSSM, Deep Structured Semantic Model."""
import typing

import torch
import torch.nn.functional as F

from ailib.models.base_model import BaseModel
from ailib.models.base_model_param import BaseModelParam
import torch
import torch.nn as nn

from ailib.param.param import Param
from ailib.param import hyper_spaces
from ailib.param.param_table import ParamTable
from ailib.tools.utils_name_parse import parse_activation
from ailib.modules.bert_module import BertModule

class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=False, with_multi_layer_perceptron=False):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self['model_name'] = "BERTMatch"
        self.add(Param(name='pretrained_model_path', value='albert_chinese_tiny',
                        desc='the path or name of the pretrain model'))
        self.add(Param(name='pretrained_model_out_dim', value=768,
                        desc='the dim of the pretrain model'))
        self.add(Param(name='dropout_rate', value=0.0,
            hyper_space=hyper_spaces.quniform(low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))

class Model(BaseModel):
    """
    BERT semantic model.

    Examples:
        >>> model_param = ModelParam()
        >>> model_param['task'] = ranking_task
        >>> model_param['pretrained_model_path'] = 'albert_chinese_tiny'
        >>> model_param['pretrained_model_out_dim'] = 312
        >>> model_param.to_frame()
        >>> model = Model(model_param.to_config())

    """
    def __init__(self, config):
        """
        BERT arthitecture.
        """
        super().__init__()
        self.config = config
        self.bert = BertModule(self.config.pretrained_model_path)
        self.dropout = nn.Dropout(p=self.config.dropout_rate)
        self.out = self._make_output_layer(self.config.pretrained_model_out_dim)

    def forward(self, inputs):
        """Forward."""
        input_left, input_right = inputs['text_left'], inputs['text_right']
        bert_output = self.bert(input_left, input_right)[1]
        out = self.out(self.dropout(bert_output))
        return out
