from ailib.models.base_model import BaseModule
import torch, torch.nn.functional as F
from torch import ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor
from torch import nn, optim, as_tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, TensorDataset
from torch.nn.utils import weight_norm, spectral_norm


class FastTextClassifier(BaseModule):

    def __init__(self, vocab_size, embedding_dim=256, n_classes=2, hidden_size=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, n_classes)
        )

    def init_weights(self, pretrained_word_vectors=None, is_static=False):
        if pretrained_word_vectors:
            self.embedding.weight = nn.Parameter(torch.from_numpy(pretrained_word_vectors).float())
        else:
            nn.init.uniform_(self.embedding.weight, -1.0, 1.0)

        if is_static:
            self.embedding.weight.requires_grad = False

    def forward(self, inputs):
        out = self.embedding(inputs).unsqueeze(1) # (B,1,T,D)
        out = self.fc(torch.mean(out, dim=1)) # (batch, n_classes)
        return out
