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
from ailib.text.basic_data import ch_en_punctuation
from transformers import AutoModel

class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=False, with_multi_layer_perceptron=False):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self['model_name'] = "ColBERT"
        self.add(Param(name='pretrained_model_path', value='albert_chinese_tiny', desc='the path or name of the pretrain model'))
        self.add(Param(name='pretrained_model_out_dim', value=312, desc='the dim of the pretrain model'))
        self.add(Param(name='linear_dim', value=128, desc="linear_dim."))
        self.add(Param(name='query_maxlen', value=0.0, desc="query_maxlen."))
        self.add(Param(name='doc_maxlen', value=0.0, desc="doc_maxlen."))
        self.add(Param(name='similarity_metric', value='cosine', desc="similarity_metric."))
        self.add(Param(name='skip_ids', value={}, desc="the document mask ids like punctuation and any other things. will ignore mask_punctuation option"))
        self.add(Param(name='mask_punctuation', value=True, desc="whether mask document punctuation."))
        self.add(Param(name='tokenizer', value=None, desc="the tokenizer used to tokenize punctuation in doc."))


class Model(BaseModel):
    """
    ColBERT model.

    Examples:
        >>> from ailib.tasks.ranking import RankingTask
        >>> from ailib.loss_function import RankHingeLoss
        >>> model_param = ModelParam()
        >>> ranking_task = RankingTask(losses= RankHingeLoss())
        >>> model_param['task'] = ranking_task
        >>> model_param['tokenizer'] = tokenizer
        >>> model_param['pretrained_model_path'] = 'albert_chinese_tiny'
        >>> model_param['pretrained_model_out_dim'] = 768
        >>> model_param['similarity_metric'] = "l2"
        >>> model = Model(model_param.to_config())
    """
    def __init__(self, config):
        """
        ColBERT arthitecture.
        """
        super().__init__()
        self.config = config
        self.bert = AutoModel.from_pretrained(self.config.pretrained_model_path)
        self.linear = nn.Linear(self.config.pretrained_model_out_dim, self.config.linear_dim)
        self.skip_ids = self.config.skip_ids
        if not self.skip_ids and self.config.mask_punctuation:
            self.skip_ids = {w: True for symbol in ch_en_punctuation for w in [symbol]+self.config.tokenizer(symbol)}

    def mask(self, input_ids):
        mask = [[(x not in self.skip_ids) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask

    def query(self, input_ids):
        Q = self.bert(input_ids)[0]
        Q = self.linear(Q)
        #[batch_size, input_len, hidden_dim]
        Q = torch.nn.functional.normalize(Q, p=2, dim=2)
        return Q

    def doc(self, input_ids):
        D = self.bert(input_ids)[0]
        D = self.linear(D)
        mask = torch.tensor(self.mask(input_ids), device=input_ids.device).unsqueeze(2).float()
        D = D * mask
        #[batch_size, input_len, hidden_dim]
        D = torch.nn.functional.normalize(D, p=2, dim=2)
        return D

    def score(self, Q, D):
        if self.config.similarity_metric == 'cosine':
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)

        assert self.config.similarity_metric == 'l2'
        return (-1.0 * ((Q.unsqueeze(2) - D.unsqueeze(1))**2).sum(-1)).max(-1).values.sum(-1)

    def forward(self, inputs):
        """Forward."""
        input_left, input_right = inputs['text_left'], inputs['text_right']
        return self.score(self.query(input_left), self.doc(input_right))

    