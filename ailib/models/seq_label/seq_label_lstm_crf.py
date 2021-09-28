from ailib.models.base_model import BaseModel
import torch, torch.nn.functional as F
from torch import ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor
from torch import nn, optim, as_tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, TensorDataset
from torch.nn.utils import weight_norm, spectral_norm
from torch.nn import LayerNorm
from ailib.modules.crf import CRF
from ailib.models.base_model_param import BaseModelParam
from ailib.param.param import Param

class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=True, with_multi_layer_perceptron=False):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self['model_name'] = "LSTM-CRF"
        self['learning_rate'] = 1e-3
        self.add(Param(
            name='embed_size',
            value='300',
            desc='embed size'
        ))
        self.add(Param(
            name='dropout', value=0, desc='lstm dropout rate'
        ))
        self.add(Param(
            name='hidden_size', value=256, desc='lstm hidden size'
        ))
        self.add(Param(
            name='num_layers', value=2, desc='nums of lstm layer'
        ))
        self.add(Param(
            name='bidirectional', value=True, desc='whether use bidirection lstm'
        ))
        self.add(Param(
            name='sos_tag_id', value=0, desc='end of sentence tag id'
        ))
        self.add(Param(
            name='eos_tag_id', value=1, desc='start of sentence tag id'
        ))
        self.add(Param(
            name='tag_vocab_size', value=0, desc='tag vocab size'
        ))

class SpatialDropout(nn.Dropout2d):
    def __init__(self, p=0.6):
        super(SpatialDropout, self).__init__(p=p)

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class Model(BaseModel):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.START_TAG_ID = config.tag_vocab_size
        self.STOP_TAG_ID = config.tag_vocab_size + 1
        self.tagset_size = config.tag_vocab_size + 2
        self.embedding = self._make_default_embedding_layer()
        self.bilstm = nn.LSTM(input_size=config.embedding_output_dim ,hidden_size=config.hidden_size,
                              batch_first=True, num_layers=config.num_layers, dropout=config.dropout,
                              bidirectional=config.bidirectional)
        self.dropout = SpatialDropout(config.dropout)
        self.bilstm_output_size = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        self.layer_norm = LayerNorm(self.bilstm_output_size)
        self.classifier = nn.Linear(self.bilstm_output_size, self.tagset_size)
        self.crf = CRF(self.START_TAG_ID, self.STOP_TAG_ID, tagset_size=self.tagset_size)

    def forward(self, inputs):
        input_ids = inputs['input_ids']
        embs = self.embedding(input_ids)
        embs = self.dropout(embs)
        sequence_output, _ = self.bilstm(embs)
        sequence_output= self.layer_norm(sequence_output)
        features = self.classifier(sequence_output)
        return features

    def loss(self, inputs, outputs, targets):
        return self.crf.calculate_loss(outputs, inputs['input_lengths'], targets['target'])

    def features2label(self, features, input_lengths):
        tags, confidences = self.crf.obtain_labels(features, input_lengths)
        return tags

    def evaluate(self, inputs, targets):
        input_ids, input_lengths = inputs['input_ids'], inputs['input_lengths']
        features = self.forward(inputs)
        loss = self.loss(inputs, features, targets)
        tags = self.features2label(features, input_lengths)
        return tags, loss

    def predict(self, inputs):
        input_ids, input_lengths = inputs['input_ids'], inputs['input_lengths']
        features = self.forward(inputs)
        tags = self.features2label(features, input_lengths)
        return tags

    def optimizer(self):
        return optim.Adam