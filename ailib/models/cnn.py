from ailab.models.base_model import BaseModule
import torch, torch.nn.functional as F
from torch import ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor
from torch import nn, optim, as_tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, TensorDataset
from torch.nn.utils import weight_norm, spectral_norm


class TextCNNClassifier(BaseModule):

    def __init__(self, vocab_size, embedding_dim=256, n_class=2, kernel_dim=100, kernel_sizes=(3, 4, 5), dropout_p=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_dim, (K, embedding_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(len(kernel_sizes) * kernel_dim, n_class)

    def init_weights(self, pretrained_word_vectors=None, is_static=False):
        if pretrained_word_vectors:
            self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        else:
            nn.init.uniform_(self.embedding.weight, -1.0, 1.0)

        if is_static:
            self.embedding.weight.requires_grad = False

    def forward(self, inputs):
        inputs = self.embedding(inputs).unsqueeze(1) # (B,1,T,D)
        inputs = [F.relu(conv(inputs)).squeeze(3) for conv in self.convs] #[(N,Co,W), ...]*len(Ks)
        inputs = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in inputs] #[(N,Co), ...]*len(Ks)

        concated = torch.cat(inputs, 1)

        concated = self.dropout(concated) # (N,len(Ks)*Co)
        out = self.fc(concated)
        return out

# class TextCNN(BaseModule):

#     def conv_and_pool(self, x, conv):
#         x = F.relu(conv(x)).squeeze(-1) #[n,c,w]
#         x = F.max_pool1d(x, x.size(-1)).squeeze(-1) #[n,c]
#         return x

#     def __init__(self, embedding_num, embedding_dim=256, kenel_num=16, each_kenel_size = [2,3,4,5], dropout_rate = 0.2):
#         super().__init__()
#         self.embedding = nn.Embedding(embedding_num, embedding_dim)
#         nn.init.uniform_(self.embedding.weight, -1.0, 1.0)
#         self.each_kenel_size = each_kenel_size
#         self.convs = nn.ModuleList(nn.Conv2d(1, kenel_num, (k, embedding_dim)) for k in self.each_kenel_size)
#         self.linear = nn.Linear(kenel_num * len(self.each_kenel_size), 2)
#         self.dropout = nn.Dropout(dropout_rate)

#     def forward(self, x):
#         x = self.embedding(x.long())
#         x = x.unsqueeze(-3) #[n,c,w,h]
#         x = [self.conv_and_pool(x, conv) for conv in self.convs]
#         x = torch.cat(x, 1)
#         x = self.dropout(x)
#         x = self.linear(x)
#         return x
