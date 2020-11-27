from ailib.models.base_model import BaseModule
import torch, torch.nn.functional as F
from torch import ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor
from torch import nn, optim, as_tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, TensorDataset
from torch.nn.utils import weight_norm, spectral_norm
from torch.nn import LayerNorm
from ailib.modules.crf import CRF

class Config(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'LSTM-CRF'
        self.embedding_pretrained = None                                # 预训练词向量
        self.dropout = 0.1                                              # 随机失活
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.n_tag = 0                                                  # tag表大小，在运行时赋值, 不包括START_TAG,STOP_TAG,PAD_TAG
        self.padding_idx = 0                                            # embedding层padding_idx
        self.learning_rate = 1e-3                                       # 学习率
        self.embed_size = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.hidden_size = 256                                          # lstm隐藏层
        self.num_layers = 2                                             # lstm层数
        self.bidirectional = True                                       # 是否双向lstm


class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class Model(BaseModule):

    def __init__(self, config):
        super().__init__()
        self.START_TAG_ID = config.n_tag
        self.STOP_TAG_ID = config.n_tag + 1
        self.tagset_size = config.n_tag + 2
        self.emebdding_size = config.embed_size
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_size, padding_idx=config.padding_idx)

        self.bilstm = nn.LSTM(input_size=config.embed_size ,hidden_size=config.hidden_size,
                              batch_first=True, num_layers=config.num_layers,dropout=config.dropout,
                              bidirectional=config.bidirectional)
        self.dropout = SpatialDropout(config.dropout)
        self.hidden_input_size = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        self.layer_norm = LayerNorm(self.hidden_input_size)
        self.classifier = nn.Linear(self.hidden_input_size, self.tagset_size)
        self.crf = CRF(self.START_TAG_ID, self.STOP_TAG_ID, tagset_size=self.tagset_size)

    def forward(self, inputs_ids):
        embs = self.embedding(inputs_ids)
        embs = self.dropout(embs)
        seqence_output, _ = self.bilstm(embs)
        seqence_output= self.layer_norm(seqence_output)
        features = self.classifier(seqence_output)
        return features

    def loss(self, input_ids, input_lens, input_tags):
        features = self.forward(input_ids)
        return self.crf.calculate_loss(features, input_lens, input_tags)

    def evaluate(self, input_ids, input_lens, input_tags):
        features = self.forward(input_ids)
        loss = self.crf.calculate_loss(features, input_lens, input_tags)
        tags, confidences = self.crf.obtain_labels(features, input_lens)
        return tags, confidences, loss

    def predict(self, input_ids, input_lens):
        features = self.forward(input_ids)
        tags, confidences = self.crf.obtain_labels(features, input_lens)
        return tags, confidences

    def optimizer(self):
        return optim.Adam