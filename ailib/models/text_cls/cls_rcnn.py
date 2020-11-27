from ailib.models.base_model import BaseModule
import torch, torch.nn.functional as F
from torch import ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor
from torch import nn, optim, as_tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, TensorDataset
from torch.nn.utils import weight_norm, spectral_norm

class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'TextRCNN'
        self.embedding_pretrained = None                                # 预训练词向量
        self.dropout = 1.0                                              # 随机失活
        self.n_classes = 2                                              # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.padding_idx = 0                                            # embedding层padding_idx
        self.learning_rate = 1e-3                                       # 学习率
        self.embed_size = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.bidirectional = True                                       # 是否双向lstm

class Model(BaseModule):
    '''Recurrent Convolutional Neural Networks for Text Classification'''
    def __init__(self, config):
        super().__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_size, padding_idx=config.padding_idx)
        self.lstm = nn.LSTM(config.embed_size, config.hidden_size, config.num_layers,
                            bidirectional=config.bidirectional, batch_first=True, dropout=config.dropout)
        hidden_input_size = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        self.fc_middle = nn.Linear(hidden_input_size + config.embed_size, hidden_input_size)
        self.fc = nn.Linear(hidden_input_size, config.n_classes)

    def forward(self, inputs):
        embed = self.embedding(inputs)
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = self.fc_middle(out)
        out = torch.tanh(out)
        out = out.permute(0, 2, 1)
        out = F.max_pool1d(out, out.size(2)).squeeze(dim=2)
        out = self.fc(out)
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
        raise optim.Adam