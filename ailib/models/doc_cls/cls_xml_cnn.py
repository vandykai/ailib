import torch
import torch.nn as nn
import torch.nn.functional as F
from ailib.models.base_model import BaseModule

class Config():
    """配置参数"""
    def __init__(self):
        self.model_name = "HAN"
        self.learning_rate = 0.01
        self.embedding_pretrained = None
        self.n_vocab = 0
        self.dropout = 0.5
        self.embed_size = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.padding_idx = 0
        self.n_classes = 0
        self.num_filters = 100
        self.num_bottleneck_hidden = 512
        self.dynamic_pool_length = 32

class Model(BaseModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_bottleneck_hidden = config.num_bottleneck_hidden
        self.dynamic_pool_length = config.dynamic_pool_length
        self.ks = 3 # There are three conv nets here
        self.input_channel = 1
        rand_embed_init = torch.Tensor(config.n_vocab, config.embed_size).uniform_(-0.25, 0.25)
        self.embedding = nn.Embedding.from_pretrained(rand_embed_init, freeze=False)

        ## Different filter sizes in xml_cnn than kim_cnn
        self.conv1 = nn.Conv2d(self.input_channel, config.num_filters, (2, config.embed_size), padding=(1,0))
        self.conv2 = nn.Conv2d(self.input_channel, config.num_filters, (4, config.embed_size), padding=(3,0))
        self.conv3 = nn.Conv2d(self.input_channel, config.num_filters, (8, config.embed_size), padding=(7,0))

        self.dropout = nn.Dropout(config.dropout)
        self.bottleneck = nn.Linear(self.ks * config.num_filters * self.dynamic_pool_length, self.num_bottleneck_hidden)
        self.fc = nn.Linear(self.num_bottleneck_hidden, config.n_classes)

        self.pool = nn.AdaptiveMaxPool1d(self.dynamic_pool_length) #Adaptive pooling

    def forward(self, x):
        x = self.embedding(x) # (batch, sent_len, embed_dim)
        x = x.unsqueeze(1) # (batch, channel_input, sent_len, embed_dim)
        print(x.shape)
        print(F.relu(self.conv2(x)).squeeze(3).shape)
        x = [F.relu(self.conv1(x)).squeeze(3), F.relu(self.conv2(x)).squeeze(3), F.relu(self.conv3(x)).squeeze(3)]
        x = [self.pool(i).squeeze(2) for i in x]
        print(self.pool(x[0]).shape)
        # (batch, channel_output) * ks
        x = torch.cat(x, 1) # (batch, channel_output * ks)
        print(x.shape)
        x = F.relu(self.bottleneck(x.view(-1, self.ks * self.config.num_filters * self.dynamic_pool_length)))
        x = self.dropout(x)
        x = self.fc(x) # (batch, target_size)
        return x
