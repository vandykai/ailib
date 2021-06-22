# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

from ailib.tools import utils_cache, utils_onnx


class MemoryAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, qdim=None, kdim=None, vdim=None, qk_embed_dim=None, num_heads=1, dropout=0., bias=True):
        super().__init__()
        self.qdim = qdim
        self.kdim = kdim if kdim is not None else qdim
        self.qk_same_dim = self.kdim == qdim
        self.qk_embed_dim = qk_embed_dim if qk_embed_dim is not None else qdim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = self.qk_embed_dim  // num_heads
        assert self.head_dim * num_heads == self.qk_embed_dim, "qk_embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(self.qdim, self.qk_embed_dim , bias=bias)
        self.k_proj = nn.Linear(self.kdim, self.qk_embed_dim , bias=bias)

        self.out_proj = nn.Linear(num_heads*vdim, num_heads*vdim, bias=bias)
        self.reset_parameters()
        self.onnx_trace = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        if self.qk_same_dim:
            # Empirically observed the convergence to be much better with
            # the scaled initialization
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(
        self,
        query, key, value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        before_softmax=False,
        need_head_weights=False,
    ):
        """Input shape: Time x Batch x Channel

        Args:
            - query: :math:`(tgt_len, bsz, embed_dim)` where tgt_len is the target sequence length, bsz is the batch size, embed_dim is
                the embedding dimension.
            - key: :math:`(src_len, bsz, kdim)`, where src_len is the source sequence length, bsz is the batch size, kdim is
                the embedding dimension.
            - value: :math:`(src_len, bsz, vdim)` where src_len is the source sequence length, bsz is the batch size, vdim is
                the embedding dimension.
            - key_padding_mask (BoolTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by True.
            - need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            - attn_mask 2D mask :math:`(tgt_len, src_len)` where tgt_len is the target sequence length, src_len is the source sequence length. (default: None).
            - before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            - need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        Outputs:
        if before_softmax:
            - attn_weights: :math:`(bsz * num_heads, tgt_len, src_len)`
            - v: :math:`(bsz * num_heads, src_len, head_dim)`
        else:
            - attn_output: :math:`(tgt_len, bsz, embed_dim)` where tgt_len is the target sequence length, bsz is the batch size,
                embed_dim is the embedding dimension.
            - attn_output_weights: :math:`(bsz, num_heads, tgt_len, src_len)` if need_head_weights else (bsz, tgt_len, src_len) if need_weights, else None
                where bsz is the batch size, tgt_len is the target sequence length, src_len is the source sequence length.
        """
        if need_head_weights:
            need_weights = True

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.qdim 
        assert list(query.size()) == [tgt_len, bsz, self.qdim ]

        q = self.q_proj(query)
        k = self.k_proj(key)
        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        # [bsz, src_len, vdim]
        v = value.transpose(0, 1)
        src_len = k.size(1)

        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            if attn_mask.dim() != 3:
                attn_mask = attn_mask.unsqueeze(0)
                if self.onnx_trace:
                    attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            if attn_mask.dtype == torch.bool:
                attn_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_weights += attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),
                float('-inf'),
            )
            attn_weights = attn_weights.view(bsz*self.num_heads, tgt_len, src_len)

        if before_softmax:
            return attn_weights

        attn_weights_float = utils_onnx.softmax(attn_weights, dim=-1, onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights_float.type_as(attn_weights), p=self.dropout, training=self.training)
        if attn_mask is not None and key_padding_mask is not None and attn_mask.dtype == torch.bool:
            attn_probs = attn_probs.view(bsz, self.num_heads, tgt_len, src_len)
            attn_probs = attn_probs.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2) + attn_mask.view(bsz, self.num_heads, tgt_len, src_len), 0)
            attn_probs = attn_probs.view(bsz * self.num_heads, tgt_len, src_len)
        attn_probs = attn_probs.view(bsz, self.num_heads, tgt_len, src_len).transpose(0, 1)
        # [num_heads, bsz, tgt_len, src_len]*[bsz, src_len, vdim] -> [num_heads, bsz, tgt_len, vdim]
        attn = torch.matmul(attn_probs, v)
        # [bsz, num_heads, tgt_len, vdim]
        attn = attn.permute(2, 1, 0, 3).reshape(tgt_len, bsz, -1)
        attn = self.out_proj(attn)
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads, tgt_len, src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        else:
            attn_weights = None

        return attn, attn_weights

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights
