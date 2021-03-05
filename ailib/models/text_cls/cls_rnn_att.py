from ailib.models.base_model import BaseModel
import torch, torch.nn.functional as F
from torch import ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor
from torch import nn, optim, as_tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, TensorDataset
from torch.nn.utils import weight_norm, spectral_norm

class ModelConfig(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'TextRNN_Att'
        self.embedding_pretrained = None                                # 预训练词向量
        self.dropout = 0.5                                              # 随机失活
        self.n_classes = 2                                              # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.padding_idx = 0                                            # embedding层padding_idx
        self.learning_rate = 1e-3                                       # 学习率
        self.embed_size = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.hidden_size = 128                                          # lstm隐藏层
        self.hidden_size2 = 64
        self.num_layers = 2                                             # lstm层数
        self.bidirectional = True                                       # 是否双向lstm

class Model(BaseModel):
    '''Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification'''
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_size, padding_idx=config.padding_idx)

        self.lstm = nn.LSTM(config.embed_size, config.hidden_size, config.num_layers,
                            bidirectional=config.bidirectional, batch_first=True, dropout=config.dropout)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.zeros(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2 if config.bidirectional else config.hidden_size, config.hidden_size2)
        self.fc = nn.Linear(config.hidden_size2, config.n_classes)

    def forward(self, inputs):
        emb = self.embedding(inputs)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out

    def loss_function(self):
        return nn.CrossEntropyLoss

    def optimizer(self):
        raise optim.Adam