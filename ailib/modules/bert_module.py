"""Bert module."""
import typing

import torch
import torch.nn as nn
from transformers import AutoModel
from transformers.models.electra.modeling_electra import ElectraSelfAttention


class BertModule(nn.Module):
    """
    Bert module.

    BERT (from Google) released with the paper BERT: Pre-training of Deep
    Bidirectional Transformers for Language Understanding by Jacob Devlin,
    Ming-Wei Chang, Kenton Lee and Kristina Toutanova.

    :param pretrained_model_path: String, supported model can be referred
        https://huggingface.co/transformers/model_doc/auto.html.

    """

    def __init__(self, pretrained_model_path: str = 'bert-base-uncased'):
        """:class:`BertModule` constructor."""
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model_path)

    def forward(self, x, y=None):
        """Forward."""
        input_ids = torch.cat((x, y), dim=-1)
        token_type_ids = torch.cat((
            torch.zeros_like(x),
            torch.ones_like(y)), dim=-1).long()
        attention_mask = (input_ids != 0)
        return self.bert(input_ids=input_ids, token_type_ids=token_type_ids,
                        attention_mask=attention_mask)