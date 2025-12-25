import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat
from ailib.modules.transformer import Transformer

import torch, torch.nn.functional as F
from torch import ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor
from torch import nn, optim, as_tensor
from ailib.models.base_model_param import BaseModelParam
from ailib.param.param import Param

class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=True, with_multi_layer_perceptron=False):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self['model_name'] = "MAE"
        self['learning_rate'] = 1e-3

        self.add(Param(name='masking_ratio', value=0.75, desc="masking_ratio."))
        self.add(Param(name='emb_dropout', value=0, desc='embedding dropout rate'))
        self.add(Param(name='max_input_length', value=128, desc="max length for each input."))

        self.add(Param(name='encoder_embed_dim', value=300, desc='encoder embedding size'))
        self.add(Param(name='encoder_depth', value=6, desc="encoder transformer blocks."))
        self.add(Param(name='encoder_heads', value=8, desc="encoder transformer heads."))
        self.add(Param(name='encoder_dim_head', value=64, desc="encoder transformer head dim."))
        self.add(Param(name='encoder_mlp_dim', value=128, desc="encoder transformer feedforward dim."))
        self.add(Param(name='encoder_attn_dropout', value=0., desc="encoder transformer dropout rate."))

        self.add(Param(name='decoder_embed_dim', value=300, desc='decoder embedding size'))
        self.add(Param(name='decoder_depth', value=1, desc="decoder transformer blocks."))
        self.add(Param(name='decoder_heads', value=8, desc="decoder transformer heads."))
        self.add(Param(name='decoder_dim_head', value=64, desc="decoder transformer head dim."))
        self.add(Param(name='decoder_mlp_dim', value=128, desc="encoder transformer feedforward dim."))
        self.add(Param(name='decoder_attn_dropout', value=0., desc="decoder transformer dropout rate."))



class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.masking_ratio > 0 and config.masking_ratio < 1, 'masking ratio must be kept between 0 and 1'
        self.masking_ratio = config.masking_ratio

        # self.to_patch_embedding = nn.Sequential(
        #     nn.Linear(patch_dim, dim),
        # )

        # encoder part
        self.encoder_pos_embedding = nn.Parameter(torch.randn(1, config.max_input_length, config.encoder_embed_dim))
        self.encoder_attention= Transformer(dim=config.encoder_embed_dim, depth=config.encoder_depth, heads=config.encoder_heads, 
                                        dim_head=config.encoder_dim_head, mlp_dim=config.encoder_mlp_dim, dropout=config.encoder_attn_dropout)

        # common
        self.enc_to_dec = nn.Linear(config.encoder_embed_dim, config.decoder_embed_dim) if config.encoder_embed_dim != config.decoder_embed_dim else nn.Identity()
        self.dropout = nn.Dropout(config.emb_dropout)

        # decoder part
        self.mask_token = nn.Parameter(torch.randn(config.decoder_embed_dim))
        self.decoder_pos_emb = nn.Embedding(config.max_input_length, config.decoder_embed_dim)
        self.decoder_attention = Transformer(dim=config.decoder_embed_dim, depth=config.decoder_depth, heads=config.decoder_heads, 
                                        dim_head=config.decoder_dim_head, mlp_dim=config.decoder_mlp_dim, dropout=config.decoder_attn_dropout)


        self.to_point = nn.Linear(config.decoder_embed_dim, 1)

    def encoder(self, tokens):
        _, num_tokens, *_ = tokens.shape
        tokens += self.encoder_pos_embedding[:, :num_tokens]
        # attend encoder tokens with transformer
        tokens = self.encoder_attention(tokens)
        return tokens

    def forward(self, inputs):
        tokens = inputs['tokens']
        device = tokens.device
        bsz, num_tokens, *_ = tokens.shape

        tokens += self.encoder_pos_embedding[:, :num_tokens]

        # calculate of tokens needed to be masked, and get random indices, dividing it up for mask vs unmasked
        num_masked = int(self.masking_ratio * num_tokens)
        rand_indices = torch.rand(bsz, num_tokens, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded
        batch_range = torch.arange(bsz, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]


        # get the tokens to be masked for the final reconstruction loss
        masked_tokens = inputs['tokens'][batch_range, masked_indices]

        # attend encoder tokens with transformer
        encoded_tokens = self.encoder_attention(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder
        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above
        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = bsz, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder
        decoder_tokens = torch.cat((decoder_tokens, mask_tokens), dim = 1)
        decoded_tokens = self.decoder_attention(decoder_tokens)

        # splice out the mask tokens and project to pixel values
        mask_tokens = decoded_tokens[:, -num_masked:]
        pred_token_values = self.to_point(mask_tokens)

        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_token_values, masked_tokens)
        return recon_loss