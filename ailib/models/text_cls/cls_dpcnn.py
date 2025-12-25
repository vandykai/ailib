from ailib.models.base_model import BaseModel
import torch, torch.nn.functional as F
from torch import ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor
from torch import nn, optim, as_tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, TensorDataset
from torch.nn.utils import weight_norm, spectral_norm

class ModelConfig(object):

    """配置参数"""
    def __init__(self):
        self.model_name = 'DPCNN'
        self.embedding_pretrained = None                                # 预训练词向量
        self.dropout = 0.5                                              # 随机失活
        self.n_classes = 2                                              # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.padding_idx = 0                                            # embedding层padding_idx
        self.learning_rate = 1e-3                                       # 学习率
        self.embed_size = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.num_filters = 250                                          # 卷积核数量(channels数)

class Model(BaseModel):
    '''
    Deep Pyramid Convolutional Neural Networks for Text Categorization
    model input need to feed with fix length
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_size, padding_idx=config.padding_idx)

        # region embedding
        self.region_embedding = nn.Sequential(
            nn.Conv1d(config.embed_size, config.num_filters,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=config.num_filters),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        self.conv_block = nn.Sequential(
            nn.BatchNorm1d(num_features=config.num_filters),
            nn.ReLU(),
            nn.Conv1d(config.num_filters, config.num_filters,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=config.num_filters),
            nn.ReLU(),
            nn.Conv1d(config.num_filters, config.num_filters,
                      kernel_size=3, padding=1),
        )
        self.seq_len = config.pad_size
        resnet_block_list = []
        while (self.seq_len > 2):
            resnet_block_list.append(ResnetBlock(config.num_filters))
            self.seq_len = self.seq_len // 2
        self.resnet_layer = nn.Sequential(*resnet_block_list)
        self.fc = nn.Sequential(
            nn.Linear(config.num_filters*self.seq_len, config.n_classes),
            nn.BatchNorm1d(config.n_classes),
            nn.ReLU(inplace=True),
            nn.Dropout(config.dropout),
            nn.Linear(config.n_classes, config.n_classes)
        )

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = x.permute(0, 2, 1)  # (batch, w_emb, s_len)
        x = self.region_embedding(x)  # (batch, featrue_map_num, s_len)
        x = self.conv_block(x)  # (batch, featrue_map_num, s_len)
        x = self.resnet_layer(x)  # (batch, featrue_map_num, o_last_resnet_size)
        x = x.permute(0, 2, 1)  # (batch, o_last_resnet_size, featrue_map_num)
        x = x.contiguous().view(x.size(0), -1)  # (batch, featrue_map_num * o_last_resnet_size)
        out = self.fc(x)  # (batch, n_classes)
        out_prob = F.softmax(out, dim=1)  # (batch, n_classes)
        return out_prob

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

class ResnetBlock(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d(padding=(0, 1), value=0),
            nn.MaxPool1d(kernel_size=3, stride=2)
        )
        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_features=num_features),
            nn.ReLU(),
            nn.Conv1d(num_features, num_features,
                      kernel_size=3, padding=1),
            nn.BatchNorm1d(num_features=num_features),
            nn.ReLU(),
            nn.Conv1d(num_features, num_features,
                      kernel_size=3, padding=1),
        )

    def forward(self, x):
        x_shortcut = self.maxpool(x)    # shortcut connection / skip connection
        x = self.conv(x_shortcut)
        x = x + x_shortcut
        return x
