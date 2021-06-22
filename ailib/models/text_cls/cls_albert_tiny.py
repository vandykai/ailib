from ailib.models.base_model import BaseModel
import torch, torch.nn.functional as F
from torch import nn, optim
from ailib.models.base_model_param import BaseModelParam
from ailib.param.param import Param
from ailib.param import hyper_spaces
from transformers import AlbertModel

class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=True, with_multi_layer_perceptron=False):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self['model_name'] = "AlbertTiny"
        self['learning_rate'] = 3e-5
        self.add(Param(name='pretrained_model_path', value="~/pretrain_model/HuggingFace/albert_chinese_tiny/voidful/",
                         desc="albert tiny预训练模型路径."))

class Model(BaseModel): 
    '''
    Albert Tiny Model for Sentence Classification
    '''
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.sentence_encoder = AlbertModel.from_pretrained(
            self.config.pretrained_model_path, num_labels=self.config.task.num_classes)
        self.out = self._make_output_layer(312)

    def forward(self, inputs):
        # Indices of input sequence tokens in the vocabulary.
        input_ids = inputs["input_ids"]
        # Mask to avoid performing attention on padding token indices. Mask values selected in [0, 1]: 1 for tokens that are not masked, 0 for tokens that are masked.
        attention_mask = inputs["attention_mask"]
        # Segment token indices to indicate first and second portions of the inputs. Indices are selected in [0, 1]: 0 corresponds to a sentence A token, 1 corresponds to a sentence B token.
        token_type_ids = inputs["token_type_ids"]

        pooler_output = self.sentence_encoder(input_ids, attention_mask, token_type_ids)[1]
        out = self.out(pooler_output)
        return out