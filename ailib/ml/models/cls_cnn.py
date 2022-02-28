import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from ailib.modules.transformer import Transformer

import torch, torch.nn.functional as F
from torch import ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor
from torch import nn, optim, as_tensor
from ailib.models.base_model_param import BaseModelParam
from ailib.models.base_model import BaseModel
from ailib.param.param import Param
from ailib.param import hyper_spaces


class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=False, with_multi_layer_perceptron=False):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self['model_name'] = "CNN"
        self['learning_rate'] = 1e-3
        self.add(Param(name='filter_sizes', value=(2, 3, 4),
                         desc="卷积核尺寸."))
        self.add(Param(
            name='num_filters', value=256,
            hyper_space=hyper_spaces.quniform(
                low=128, high=512, q=32),
            desc="卷积核数量(channels数)"))
        self.add(Param(
            name='dropout_rate', value=0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))

class Model(BaseModel): 
    '''
    Convolutional Neural Networks for Sentence Classification
    model input need to feed with fix length
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.config.num_filters, kernel_size=k) for k in self.config.filter_sizes])
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.out = self._make_output_layer(self.config.num_filters * len(self.config.filter_sizes))

    def forward(self, inputs):
        device = inputs['device']
        # [batch_size, number]
        # [N, 1, H, W]
        patches = torch.sparse_coo_tensor(inputs['indices'], inputs['values'], inputs['shape'], device=device).to_dense()
        # [N, C, H_out]*len(Ks)
        input_ids = [F.relu(conv(patches)).squeeze(3) for conv in self.convs]
        # [N, C]*len(Ks)
        input_ids = [F.max_pool1d(x, x.size(2)).squeeze(2) for x in input_ids]

        concated = torch.cat(input_ids, 1)

        concated = self.dropout(concated) # (N,len(Ks)*Co)
        out = self.out(concated)
        return out

    def init_weights(self, pretrained_word_vectors=None, is_static=False):
        if pretrained_word_vectors:
            self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        else:
            nn.init.uniform_(self.embedding.weight, -1.0, 1.0)

        if is_static:
            self.embedding.weight.requires_grad = False

    def loss_function(self):
        return nn.CrossEntropyLoss

    def optimizer(self):
        return optim.Adam


