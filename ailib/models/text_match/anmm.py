"""An implementation of aNMM Model."""
import typing
import torch
import torch.nn as nn
from ailib.models.base_model import BaseModel
from ailib.models.base_match_model_param import BaseModelParam
from ailib.modules.attention import Attention
from ailib.modules.matching import Matching
from ailib.param.param import Param
from ailib.param import hyper_spaces
from ailib.param.param_table import ParamTable
from ailib.tools.utils_name_parse import parse_activation

class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=True, with_multi_layer_perceptron=False):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self.add(Param(name='model_name', value="aNMM",
                         desc="model name"))
        self.add(Param(name='mask_value', value=0,
                         desc="The value to be masked from inputs."))
        self.add(Param(name='num_bins', value=200,
                         desc="Integer, number of bins."))
        self.add(Param(name='hidden_sizes', value=[100],
                         desc="Number of hidden size for each hidden layer"))
        self.add(Param(name='activation', value='relu',
                         desc="The activation function."))

        self.add(Param(
            'dropout_rate', 0.0,
            hyper_space=hyper_spaces.quniform(
                low=0.0, high=0.8, q=0.01),
            desc="The dropout rate."
        ))

class Model(BaseModel):
    """
    aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model.

    Examples:
        >>> model = aNMM()
        >>> model.params['embedding_output_dim'] = 300
        >>> model.guess_and_fill_missing_params(verbose=0)
        >>> model.build()

    """

    def __init__(self, config):
        """
        Build model structure.

        aNMM: Ranking Short Answer Texts with Attention-Based Neural Matching Model.
        """
        super().__init__()
        self.config = config
        self.embedding = self._make_default_embedding_layer()

        # QA Matching
        self.matching = Matching(matching_type='dot', normalize=True)

        # Value-shared Weighting
        activation = parse_activation(self.config.activation)
        in_hidden_size = [
            self.config.num_bins,
            *self.config.hidden_sizes
        ]
        out_hidden_size = [
            *self.config.hidden_sizes,
            1
        ]

        hidden_layers = [
            nn.Sequential(
                nn.Linear(in_size, out_size),
                activation
            )
            for in_size, out_size, in zip(
                in_hidden_size,
                out_hidden_size
            )
        ]
        self.hidden_layers = nn.Sequential(*hidden_layers)

        # Query Attention
        self.q_attention = Attention(self.config.embedding_output_dim)

        self.dropout = nn.Dropout(p=self.config.dropout_rate)

        # Build output
        self.out = self._make_output_layer(1)

    def forward(self, inputs):
        """Forward."""
        # Scalar dimensions referenced here:
        #   B = batch size (number of sequences)
        #   D = embedding size
        #   L = `input_left` sequence length
        #   R = `input_right` sequence length
        #   BI = number of bins

        # Left input and right input
        # shape = [B, L] input_left
        # shape = [B, R] input_right
        input_left, input_right = inputs['text_left'], inputs['text_right']

        # Process left and right input
        # shape = [B, L, D]
        # shape = [B, R, D]
        embed_left = self.embedding(input_left.long())
        embed_right = self.embedding(input_right.long())

        # Left and right input mask matrix
        # shape = [B, L]
        # shape = [B, R]
        left_mask = (input_left == self.config.mask_value)
        right_mask = (input_right == self.config.mask_value)

        # Compute QA Matching matrix
        # shape = [B, L, R]
        qa_matching_matrix = self.matching(embed_left, embed_right)
        qa_matching_matrix.masked_fill_(right_mask.unsqueeze(1), float(0))

        # Bin QA Matching Matrix
        B, L = qa_matching_matrix.shape[0], qa_matching_matrix.shape[1]
        BI = self.config.num_bins
        device = qa_matching_matrix.device
        qa_matching_matrix = qa_matching_matrix.view(-1)
        qa_matching_detach = qa_matching_matrix.detach()

        bin_indexes = torch.floor((qa_matching_detach + 1.) / 2 * (BI - 1.)).long()
        bin_indexes = bin_indexes.view(B * L, -1)

        index_offset = torch.arange(start=0, end=(B * L * BI), step=BI,
                                    device=device).long().unsqueeze(-1)
        bin_indexes += index_offset
        bin_indexes = bin_indexes.view(-1)

        # shape = [B, L, BI]
        bin_qa_matching = torch.zeros(B * L * BI, device=device)
        bin_qa_matching.index_add_(0, bin_indexes, qa_matching_matrix)
        bin_qa_matching = bin_qa_matching.view(B, L, -1)

        # Apply dropout
        bin_qa_matching = self.dropout(bin_qa_matching)

        # MLP hidden layers
        # shape = [B, L, 1]
        hiddens = self.hidden_layers(bin_qa_matching)

        # Query attention
        # shape = [B, L, 1]
        q_attention = self.q_attention(embed_left, left_mask).unsqueeze(-1)

        # shape = [B, 1]
        score = torch.sum(hiddens * q_attention, dim=1)
        # shape = [B, *]
        out = self.out(score)
        return out
