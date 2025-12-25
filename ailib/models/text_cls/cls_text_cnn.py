from ailib.models.base_model import BaseModel
import torch, torch.nn.functional as F
from torch import nn, optim
from torch.nn.utils import weight_norm, spectral_norm
from ailib.models.base_model_param import BaseModelParam
from ailib.param.param import Param
from ailib.param import hyper_spaces

class ModelConfig(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'TextCNN'
        self.embedding_pretrained = None                                # 预训练词向量
        self.dropout = 0.5                                              # 随机失活
        self.n_classes = 2                                              # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.padding_idx = 0                                            # embedding层padding_idx
        self.learning_rate = 1e-3                                       # 学习率
        self.embed_size = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.task = None

class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=True, with_multi_layer_perceptron=False):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self['model_name'] = "TextCNN"
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
        self.embedding = self._make_default_embedding_layer()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, self.config.num_filters, (k, self.config.embedding_output_dim)) for k in self.config.filter_sizes])
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.out = self._make_output_layer(self.config.num_filters * len(self.config.filter_sizes))

    def forward(self, inputs):
        input_ids = inputs["text"]
        # input_ids [N, L]
        # [N, 1, H, W] H=>L,W=>embed_dim
        input_ids = self.embedding(input_ids).unsqueeze(1)
        # [N, C, H_out]*len(Ks)
        input_ids = [F.relu(conv(input_ids)).squeeze(3) for conv in self.convs]
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