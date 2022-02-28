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

class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=False, with_multi_layer_perceptron=False):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self['model_name'] = "DNN"
        self['learning_rate'] = 1e-3
        self.add(Param(name='input_dim', value=16384, desc="max length for each input."))
        self.add(Param(name='hidden_layer', value=[8096, 1024, 256, 32, 4], desc='hidden layer size'))

class Model(BaseModel):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.head_net = nn.Sequential(
            nn.BatchNorm1d(config.input_dim),
            nn.Linear(config.input_dim, config.hidden_layer[0]),
            nn.BatchNorm1d(config.hidden_layer[0]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.middle_net = []

        for i in range(len(config.hidden_layer)-1):
            layers = [
                nn.Linear(config.hidden_layer[i], config.hidden_layer[i+1]),
                nn.BatchNorm1d(config.hidden_layer[i+1]),
                nn.ReLU(),
                nn.Dropout(0.1)
            ]
            self.middle_net.extend(layers)

        self.middle_net = nn.Sequential(*self.middle_net)
        
        self.out = self._make_output_layer(config.hidden_layer[-1])

    def forward(self, inputs):
        device = inputs['device']
        feature = torch.sparse_coo_tensor(inputs['indices'], inputs['values'], inputs['shape'], device=device).to_dense()
        out = self.head_net(feature)
        out = self.middle_net(out)
        out = self.out(out)
        return out