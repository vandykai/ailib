from ailib.models.base_model import BaseModule
import torch, torch.nn.functional as F
from torch import ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor
from torch import nn, optim, as_tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, TensorDataset
from torch.nn.utils import weight_norm, spectral_norm

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

class Model(BaseModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.START_TAG_ID = config.n_tag
        self.STOP_TAG_ID = config.n_tag + 1
        self.tagset_size = config.n_tag + 2
        self.hidden_input_size = config.hidden_size * 2 if config.bidirectional else config.hidden_size

        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_size, padding_idx=config.padding_idx)
        self.lstm = nn.LSTM(config.embed_size, config.hidden_size, config.num_layers,
                            bidirectional=config.bidirectional, batch_first=True, dropout=config.dropout)
        
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_input_size, self.tagset_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[self.START_TAG_ID, :] = -10000
        self.transitions.data[:, self.STOP_TAG_ID] = -10000

    def init_hidden(self, inputs):
        return (torch.randn(self.config.num_layers*(2 if self.config.bidirectional else 1), inputs.shape[0], self.config.hidden_size, device=inputs.device),
                torch.randn(self.config.num_layers*(2 if self.config.bidirectional else 1), inputs.shape[0], self.config.hidden_size, device=inputs.device))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full([feats.shape[0], self.tagset_size], -10000., device=feats.device)#.to('cuda')
        # START_TAG has all of the score.
        init_alphas[:, self.START_TAG_ID] = 0.

        # Wrap in a variable so that we will get automatic backprop
        # Iterate through the sentence
        forward_var_list = []
        forward_var_list.append(init_alphas)
        for feat_index in range(feats.shape[1]):  # -1
            gamar_r_l = torch.stack([forward_var_list[feat_index]] * feats.shape[2]).transpose(0, 1)
            # gamar_r_l = torch.transpose(gamar_r_l,0,1)
            t_r1_k = torch.unsqueeze(feats[:, feat_index, :], 1).transpose(1, 2)  # +1
            # t_r1_k = feats[:,feat_index,:].repeat(feats.shape[0],1,1).transpose(1, 2)
            aa = gamar_r_l + t_r1_k + torch.unsqueeze(self.transitions, 0)
            # forward_var_list.append(log_add(aa))
            forward_var_list.append(torch.logsumexp(aa, dim=2))
        terminal_var = forward_var_list[-1] + self.transitions[self.STOP_TAG_ID].repeat([feats.shape[0], 1])
        # terminal_var = torch.unsqueeze(terminal_var, 0)
        alpha = torch.logsumexp(terminal_var, dim=1)
        return alpha

    def _get_lstm_features(self, sentence):
        # self.hidden = self.init_hidden(sentence)
        embeds = self.embedding(sentence)
        # lstm_out, self.hidden = self.lstm(embeds)
        lstm_out, _ = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(tags.shape[0], device=tags.device)
        tags = torch.cat([torch.full([tags.shape[0],1],self.START_TAG_ID, dtype=torch.long, device=tags.device),tags], dim=1)
        for i in range(feats.shape[1]):
            feat=feats[:,i,:]
            score = score + \
                    self.transitions[tags[:,i + 1], tags[:,i]] + feat[range(feat.shape[0]),tags[:,i + 1]]
        score = score + self.transitions[self.STOP_TAG_ID, tags[:,-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((feats.shape[0], self.tagset_size), -10000., device=feats.device)#.to('cuda')
        init_vvars[:, self.START_TAG_ID] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var_list = []
        forward_var_list.append(init_vvars)
        for feat_index in range(feats.shape[1]):
            gamar_r_l = forward_var_list[feat_index].unsqueeze(1).expand(-1, feats.shape[2], -1)
            next_tag_var = gamar_r_l + self.transitions
            viterbivars_t, bptrs_t = torch.max(next_tag_var, dim=2)
            forward_var_new = viterbivars_t + feats[:,feat_index]
            forward_var_list.append(forward_var_new)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var_list[-1] + self.transitions[self.STOP_TAG_ID]
        best_tag_id = torch.argmax(terminal_var, 1)
        path_score = terminal_var[range(terminal_var.shape[0]), best_tag_id]
        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[range(bptrs_t.shape[0]), best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop().tolist()
        assert start == [self.START_TAG_ID]*feats.shape[0]  # Sanity check
        best_path.reverse()
        best_path = torch.stack(best_path).permute([1,0])
        return path_score, best_path

    def loss(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return torch.mean(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq

    def predict(self, sentence):
        _, tag_seq = self.forward(sentence)
        return tag_seq