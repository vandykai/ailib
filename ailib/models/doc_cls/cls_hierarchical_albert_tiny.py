import torch
import torch.nn.functional as F
from torch import nn
from transformers import AlbertModel
from ailib.models.base_model import BaseModule

class Config():
    """配置参数"""
    def __init__(self):
        self.model_name = "HierarchicalAlbert-tiny"
        self.dropout = 0.5
        self.num_filters = 64
        self.max_sentence_length = 128
        self.max_doc_length = 16
        self.pretrained_model_path = "albert_chinese_tiny"
        self.n_classes = 0
        self.learning_rate = 3e-5

class Model(BaseModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_channel = 1
        self.ks = 3 # There are three conv nets here
        self.sentence_encoder = AlbertModel.from_pretrained(
            config.pretrained_model_path, num_labels=config.n_classes)
        self.conv1 = nn.Conv2d(self.input_channel, config.num_filters, (3, self.sentence_encoder.config.hidden_size), padding=(2, 0))
        self.conv2 = nn.Conv2d(self.input_channel, config.num_filters, (4, self.sentence_encoder.config.hidden_size), padding=(3, 0))
        self.conv3 = nn.Conv2d(self.input_channel, config.num_filters, (5, self.sentence_encoder.config.hidden_size), padding=(4, 0))
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(self.ks * config.num_filters, config.n_classes)

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
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # (batch_size, output_channels, num_sentences)
        x = torch.cat(x, 1)  # (batch_size, channel_output * ks)

        x = self.dropout(x)
        x = self.fc(x)  # (batch_size, n_classes)

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
