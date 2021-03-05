
import torch
import torch.nn as nn
from ailib.models.base_model import BaseModel

class ModelConfig():
    """配置参数"""
    def __init__(self):
        self.model_name = "HAN"
        self.word_gru_hidden_size = 50
        self.sentence_gru_hidden_size = 50
        self.learning_rate = 0.01
        self.embedding_pretrained = None
        self.n_vocab = 0
        self.embed_size = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度, 若使用了预训练词向量，则维度统一
        self.padding_idx = 0
        self.bidirectional = True                                       # 是否双向GRU
        self.n_classes = 0

class WordRNN(BaseModel):

    def __init__(self, config):
        super().__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_size, padding_idx=config.padding_idx)
        self.embedding.weight.data.uniform_(-0.25, 0.25)
        gru_output_size = 2*config.word_gru_hidden_size if config.bidirectional else config.word_gru_hidden_size
        self.word_context_weights = nn.Parameter(torch.rand(gru_output_size, 1))
        self.word_context_weights.data.uniform_(-0.25, 0.25)

        self.GRU = nn.GRU(config.embed_size, config.word_gru_hidden_size, bidirectional=config.bidirectional)
        self.word_linear = nn.Linear(gru_output_size, gru_output_size, bias=True)


    def forward(self, x):
        # x expected to be of dimensions--> (num_words, batch_size)
        x = self.embedding(x)
        h, _ = self.GRU(x)
        x = torch.tanh(self.word_linear(h))
        x = torch.matmul(x, self.word_context_weights)
        x = x.squeeze(dim=2)
        x = torch.softmax(x.transpose(1, 0), 1)
        x = torch.mul(h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        return x

class SentenceRNN(nn.Module):

    def __init__(self, config):
        super().__init__()
        sentence_gru_output_size = 2*config.sentence_gru_hidden_size if config.bidirectional else config.sentence_gru_hidden_size
        word_gru_output_size = 2*config.word_gru_hidden_size if config.bidirectional else config.word_gru_hidden_size

        self.sentence_context_weights = nn.Parameter(torch.rand(sentence_gru_output_size, 1))
        self.sentence_context_weights.data.uniform_(-0.1, 0.1)

        self.sentence_gru = nn.GRU(word_gru_output_size, config.sentence_gru_hidden_size, bidirectional=True)
        self.sentence_linear = nn.Linear(sentence_gru_output_size, sentence_gru_output_size, bias=True)
        self.fc = nn.Linear(sentence_gru_output_size, config.n_classes)


    def forward(self,x):
        h,_ = self.sentence_gru(x)
        x = torch.tanh(self.sentence_linear(h))
        x = torch.matmul(x, self.sentence_context_weights)
        x = x.squeeze(dim=2)
        x = torch.softmax(x.transpose(1,0), 1)
        x = torch.mul(h.permute(2, 0, 1), x.transpose(1, 0))
        x = torch.sum(x, dim=1).transpose(1, 0).unsqueeze(0)
        x = self.fc(x.squeeze(0))
        return x


class Model(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.word_attention_rnn = WordRNN(config)
        self.sentence_attention_rnn = SentenceRNN(config)

    def forward(self, x,  **kwargs):
        x = x.permute(1, 2, 0) # Expected : # sentences, # words, batch size
        num_sentences = x.size(0)
        word_attentions = []
        for i in range(num_sentences):
            word_attn = self.word_attention_rnn(x[i, :, :])
            word_attentions.append(word_attn)
        word_attentions = torch.cat(word_attentions)
        return self.sentence_attention_rnn(word_attentions)