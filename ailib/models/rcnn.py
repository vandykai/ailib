from ailib.models.base_model import BaseModule
import torch, torch.nn.functional as F
from torch import ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor
from torch import nn, optim, as_tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, TensorDataset
from torch.nn.utils import weight_norm, spectral_norm


class TextRCNNClassifier(BaseModule):

    def __init__(self, vocab_size, embedding_dim=256, dropout_p=0.5, n_classes=2, bidirectional=True, batch_first=True, hidden_size=128, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers,
                            bidirectional=bidirectional, batch_first=batch_first, dropout=dropout_p)
        hidden_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc_middle = nn.Linear(hidden_input_size + embedding_dim, hidden_input_size)
        self.fc = nn.Linear(hidden_input_size, n_classes)

    def forward(self, inputs):
        embed = self.embedding(inputs.long())
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