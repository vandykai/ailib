import torch
import torch.nn.functional as F
from torch import nn
from transformers import AlbertModel
from ailib.models.base_model import BaseModel
from ailib.models.base_model_param import BaseModelParam
from ailib.param.param import Param
from ailib.param import hyper_spaces
from ailib.tools.utils_name_parse import parse_activation

# class ModelConfig():
#     """配置参数"""
#     def __init__(self):
#         self.model_name = "HierarchicalAlbert-tiny"
#         self.dropout = 0.5
#         self.num_filters = 64
#         self.max_sentence_length = 128
#         self.max_doc_length = 16
#         self.pretrained_model_path = "albert_chinese_tiny"
#         self.n_classes = 0
#         self.learning_rate = 3e-5

class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=False, with_multi_layer_perceptron=False):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self['model_name'] = "HierarchicalAlbert-tiny"
        self['learning_rate'] = 3e-5
        self.add(Param(name='num_filters', value=64,
                        desc="The number of convolution kernel."))
        self.add(Param(name='max_sentence_length', value=128,
                        desc="max length for each sentence."))
        self.add(Param(name='max_doc_length', value=64,
                        desc='max number of sentences for each doc'))
        self.add(Param(name='pretrained_model_path', value='albert_chinese_tiny',
                        desc='the path or name of the pretrain albert chinese tiny model'))
        self.add(Param(
            'dropout_rate', 0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))

class Model(BaseModel):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_channel = 1
        self.ks = 3 # There are three conv nets here
        self.sentence_encoder = AlbertModel.from_pretrained(
            config.pretrained_model_path, num_labels=config.task.num_classes)
        self.conv1 = nn.Conv2d(self.input_channel, config.num_filters, (3, self.sentence_encoder.config.hidden_size), padding=(2, 0))
        self.conv2 = nn.Conv2d(self.input_channel, config.num_filters, (4, self.sentence_encoder.config.hidden_size), padding=(3, 0))
        self.conv3 = nn.Conv2d(self.input_channel, config.num_filters, (5, self.sentence_encoder.config.hidden_size), padding=(4, 0))
        self.dropout = nn.Dropout(config.dropout_rate)
        self.out = self._make_output_layer(self.ks * config.num_filters)
        #self.fc = nn.Linear(self.ks * config.num_filters, config.n_classes)

    def forward(self, inputs):
        """
        a batch is a tensor of shape [batch_size, #file_in_commit, #line_in_file]
        and each element is a line, i.e., a bert_batch,
        which consists of input_ids, input_mask, segment_ids, label_ids
        """
        input_ids, doc_len, sentence_len_list = inputs["doc"] # (batch_size, sentences, words)
        attention_mask = torch.tensor([[[1]*length+[0]*(input_ids.shape[2]-length) for length in sentence_len] for sentence_len in sentence_len_list], device=input_ids.device)
        token_type_ids = torch.zeros_like(input_ids, device=input_ids.device)

        input_ids = input_ids.permute(1, 0, 2)  # (sentences, batch_size, words)
        attention_mask = attention_mask.permute(1, 0, 2)
        token_type_ids = token_type_ids.permute(1, 0, 2)

        x_encoded = []
        for i in range(len(input_ids)):
            x_encoded.append(self.sentence_encoder(input_ids[i], attention_mask[i], token_type_ids[i])[1])

        x = torch.stack(x_encoded)  # (sentences, batch_size, hidden_size)
        x = x.permute(1, 0, 2)  # (batch_size, sentences, hidden_size)
        x = x.unsqueeze(1)  # (batch_size, input_channels, sentences, hidden_size)

        x = [F.relu(self.conv1(x)).squeeze(3),
             F.relu(self.conv2(x)).squeeze(3),
             F.relu(self.conv3(x)).squeeze(3)]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch_size, output_channels, num_sentences)
        x = torch.cat(x, 1)  # (batch_size, channel_output * ks)

        x = self.dropout(x)
        x = self.out(x)  # (batch_size, n_classes)

        return x

class HierarchicalAlbert_old(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        input_channels = 1
        ks = 3

        self.sentence_encoder = AlbertModel.from_pretrained(
            config.pretrained_model_path, num_labels=config.num_labels)
        self.convs = [nn.Conv2d(1, config.num_filters, (k, config.embed_size)) for k in config.filter_sizes]
        self.conv1 = nn.Conv2d(input_channels,
                               config.output_channel,
                               (3, self.sentence_encoder.config.hidden_size),
                               padding=(2, 0))
        self.conv2 = nn.Conv2d(input_channels,
                               config.output_channel,
                               (4, self.sentence_encoder.config.hidden_size),
                               padding=(3, 0))
        self.conv3 = nn.Conv2d(input_channels,
                               config.output_channel,
                               (5, self.sentence_encoder.config.hidden_size),
                               padding=(4, 0))

        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(ks * config.output_channel, config.num_labels)

    def forward(self, input_ids, segment_ids=None, input_mask=None):
        """
        a batch is a tensor of shape [batch_size, #file_in_commit, #line_in_file]
        and each element is a line, i.e., a bert_batch,
        which consists of input_ids, input_mask, segment_ids, label_ids
        """
        input_ids = input_ids.permute(1, 0, 2)  # (sentences, batch_size, words)
        segment_ids = segment_ids.permute(1, 0, 2)
        input_mask = input_mask.permute(1, 0, 2)
        x_encoded = []
        for i in range(len(input_ids)):
            x_encoded.append(self.sentence_encoder(input_ids[i], input_mask[i], segment_ids[i])[1])

        x = torch.stack(x_encoded)  # (sentences, batch_size, hidden_size)
        x = x.permute(1, 0, 2)  # (batch_size, sentences, hidden_size)
        x = x.unsqueeze(1)  # (batch_size, input_channels, sentences, hidden_size)

        x = [F.relu(self.conv1(x)).squeeze(3),
             F.relu(self.conv2(x)).squeeze(3),
             F.relu(self.conv3(x)).squeeze(3)]

        if self.config.dynamic_pool:
            x = [self.dynamic_pool(i).squeeze(2) for i in x]  # (batch_size, output_channels) * ks
            x = torch.cat(x, 1)  # (batch_size, output_channels * ks)
            x = x.view(-1, self.filter_widths * self.output_channel * self.dynamic_pool_length)
        else:
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch_size, output_channels, num_sentences) * ks
            x = torch.cat(x, 1)  # (batch_size, channel_output * ks)

        x = self.dropout(x)
        logits = self.fc1(x)  # (batch_size, num_labels)

        return logits, x
