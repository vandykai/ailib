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
    def __init__(self, config,vocab_size,embedding_size,hidden_size,
                 label2id,device,drop_p = 0.1):
        super(NERModel, self).__init__()
        self.emebdding_size = config.embed_size
        self.embedding = nn.Embedding(config.n_vocab, embedding_size)
        self.bilstm = nn.LSTM(input_size=config.embed_size ,hidden_size=config.hidden_size,
                              batch_first=True, num_layers=config.num_layers,dropout=config.dropout,
                              bidirectional=config.bidirectional)
        self.dropout = SpatialDropout(config.dropout)
        self.hidden_input_size = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        self.layer_norm = LayerNorm(self.hidden_input_size)
        self.classifier = nn.Linear(self.hidden_input_size, config.hidden_size)
        self.crf = CRF(config.START_TAG_ID, config.STOP_TAG_ID, tagset_size=config.n_tag)

    def forward(self, inputs_ids, input_mask):
        embs = self.embedding(inputs_ids)
        embs = self.dropout(embs)
        embs = embs * input_mask.float().unsqueeze(2)
        seqence_output, _ = self.bilstm(embs)
        seqence_output= self.layer_norm(seqence_output)
        features = self.classifier(seqence_output)
        return features

    def forward_loss(self, input_ids, input_mask, input_lens, input_tags=None):
        features = self.forward(input_ids, input_mask)
        if input_tags is not None:
            return features, self.crf.calculate_loss(features, tag_list=input_tags, lengths=input_lens)
        else:
            return features