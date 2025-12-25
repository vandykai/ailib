from ailib.models.base_model import BaseModel
import torch, torch.nn.functional as F
from torch import nn, optim
from ailib.models.base_model_param import BaseModelParam
from ailib.param.param import Param
from ailib.param import hyper_spaces
from sentence_transformers import SentenceTransformer
from ailib.modules.multihead_attention import MultiheadAttention
from ailib.modules.memory_attention import MemoryAttention
from ailib.modules.positional_embedding import PositionalEmbedding

class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=False, with_multi_layer_perceptron=False):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self['model_name'] = "KT_Transformer"
        self['learning_rate'] = 3e-5
        self.add(Param(
            name='sentence_pretrained_model_path',
            value='quora-distilbert-multilingual',
            desc='the path or name of the multilingual duplicate questions detection pretrain model quora-distilbert-multilingual'
        ))
        self.add(Param(name='multihead_attention_num_heads', value=4, desc=''))
        self.add(Param(name='lstm_hidden_size', value=128, desc=''))
        self.add(Param(name='lstm_num_layers', value=1, desc=''))
        self.add(Param(name='lstm_dropout', value=0, desc=''))
        self.add(Param(name='lstm_bidirectional', value=False, desc=''))
        self.add(Param(name='answer_embedding_input_dim', value=3, desc=''))
        self.add(Param(name='answer_embedding_output_dim', value=4, desc=''))
        self.add(Param(name='answer_embedding_padding_idx', value=0, desc=''))
        self.add(Param(name='knowledge_embedding_input_dim', value=None, desc=''))
        self.add(Param(name='knowledge_embedding_output_dim', value=64, desc=''))
        self.add(Param(name='knowledge_embedding_padding_idx', value=0, desc=''))
        self.add(Param(name='logical_type_embedding_input_dim', value=None, desc=''))
        self.add(Param(name='logical_type_embedding_output_dim', value=8, desc=''))
        self.add(Param(name='logical_type_embedding_padding_idx', value=0, desc=''))
        self.add(Param(name='need_question_embedding', value=False, desc=''))
        self.add(Param(name='question_embedding_output_dim', value=768, desc=''))

class Model(BaseModel):

    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.need_question_embedding:
            self.question_embedding_model = SentenceTransformer(config.sentence_pretrained_model_path)
            config.question_embedding_output_dim = self.question_embedding_model.get_sentence_embedding_dimension()

        multihead_attention_input_dim = config.question_embedding_output_dim + config.knowledge_embedding_output_dim + \
            config.logical_type_embedding_output_dim
        self.multihead_attention = MemoryAttention(multihead_attention_input_dim, multihead_attention_input_dim, config.answer_embedding_output_dim, num_heads=config.multihead_attention_num_heads)
        #MultiheadAttention(embed_dim=multihead_attention_input_dim, num_heads=config.multihead_attention_num_heads, vdim=config.answer_embedding_output_dim)

        bilstm_input_size = config.question_embedding_output_dim + config.knowledge_embedding_output_dim + \
            config.logical_type_embedding_output_dim + config.multihead_attention_num_heads*config.answer_embedding_output_dim
        # self.bilstm = nn.LSTM(input_size=bilstm_input_size,hidden_size=config.lstm_hidden_size,
        #                       batch_first=False, num_layers=config.lstm_num_layers, dropout=config.lstm_dropout,
        #                       bidirectional=config.lstm_bidirectional)
        self.bilstm = nn.GRUCell(bilstm_input_size, config.lstm_hidden_size)
        self.answer_embed = nn.Embedding(
                num_embeddings=config.answer_embedding_input_dim,
                embedding_dim=config.answer_embedding_output_dim,
                padding_idx=config.answer_embedding_padding_idx
            )
        self.knowledge_embed = nn.Embedding(
                num_embeddings=config.knowledge_embedding_input_dim,
                embedding_dim=config.knowledge_embedding_output_dim,
                padding_idx=config.knowledge_embedding_padding_idx
            )
        self.logical_type_embed = nn.Embedding(
                num_embeddings=config.logical_type_embedding_input_dim,
                embedding_dim=config.logical_type_embedding_output_dim,
                padding_idx=config.logical_type_embedding_padding_idx
            )
        self.positional_embed = PositionalEmbedding(num_embeddings=1000, embedding_dim=multihead_attention_input_dim)
        self.positional_embed_out = PositionalEmbedding(num_embeddings=1000, embedding_dim=bilstm_input_size)

        bilstm_output_size = config.lstm_hidden_size * 2 if config.lstm_bidirectional else config.lstm_hidden_size
        #self.layer_norm = nn.LayerNorm(bilstm_output_size)
        self.attention_layer_norm = nn.LayerNorm(multihead_attention_input_dim)
        self.out = self._make_output_layer(bilstm_output_size)
        self.loss_function = nn.CrossEntropyLoss(ignore_index=config.answer_embedding_padding_idx)
        self.init_weights()

    def get_time_mask(self, time_stamps, leak=False):
        batch_size, seq_len = time_stamps.shape
        output, inverse_indices = torch.unique_consecutive(time_stamps, return_inverse=True)
        used_inverse_indices = inverse_indices - inverse_indices.min(dim=1)[0].unsqueeze(1)
        target_inverse_indices = torch.cat([used_inverse_indices.view(batch_size, 1, seq_len)] * seq_len, 1)
        if leak:
            attention_mask = (target_inverse_indices > used_inverse_indices.view(batch_size, seq_len, 1)).bool()
        else:
            attention_mask = (target_inverse_indices >= used_inverse_indices.view(batch_size, seq_len, 1)).bool()
        inverse_indices = torch.max(inverse_indices.min(dim=1)[0].unsqueeze(1), inverse_indices - 1)
        interval_time = (time_stamps - output[inverse_indices]) // 300
        interval_time += torch.arange(seq_len, device=time_stamps.device).unsqueeze(0).repeat(batch_size, 1)
        interval_time = torch.clamp(interval_time, min=0, max=31400)
        # [batch_size*num_heads, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.config.multihead_attention_num_heads, 1, 1).reshape(-1, seq_len, seq_len)
        # for item in attention_mask: # allow current timestep attention to self
        #     item.view(-1)[::seq_len+1]=False
        #attention_mask[:, 0, :] = False
        return interval_time, attention_mask

    def get_last_time_mask(self, time_stamps, leak=False):
        batch_size, seq_len = time_stamps.shape
        output, inverse_indices = torch.unique_consecutive(time_stamps, return_inverse=True)
        used_inverse_indices = inverse_indices - inverse_indices.min(dim=1)[0].unsqueeze(1)
        if leak:
            attention_mask = (used_inverse_indices > used_inverse_indices.view(batch_size, seq_len, 1)[:, -1:,:]).bool()
        else:
            attention_mask = (used_inverse_indices >= used_inverse_indices.view(batch_size, seq_len, 1)[:, -1:,:]).bool()

        inverse_indices = torch.max(inverse_indices.min(dim=1)[0].unsqueeze(1), inverse_indices - 1)
        interval_time = (time_stamps - output[inverse_indices]) // 300
        interval_time += torch.arange(seq_len, device=time_stamps.device).unsqueeze(0).repeat(batch_size, 1)
        interval_time = torch.clamp(interval_time, min=0, max=31400)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.config.multihead_attention_num_heads, 1, 1).reshape(-1, 1, seq_len)
        # for item in attention_mask: # allow current timestep attention to self
        #     item.view(-1)[::seq_len+1]=False
        #attention_mask[:, 0, :] = False
        # [bsz*num_heads, 1, seq_len-1)
        return interval_time, attention_mask[:, :, :-1]

    def init_weights(self):
        self.multihead_attention.reset_parameters()
        torch.nn.init.constant_(self.answer_embed.weight[0], -1)
        torch.nn.init.ones_(self.answer_embed.weight[1])

    def forward(self, inputs):
        time_stamps = inputs["time_stamps"]
        question_answers = inputs["question_answers"]
        input_lengths = inputs["input_lengths"]

        interval_time, attention_mask = self.get_time_mask(time_stamps)
        # [seq_length, batch_size, embed_dim]
        if 'questions_embeds' in inputs:
            question_embeds = inputs["questions_embeds"].transpose(0, 1)
        else:
            questions = inputs["questions"]
            question_embeds = torch.stack([self.question_embedding_model.encode(question, convert_to_tensor=True, show_progress_bar=False, device=time_stamps.device) for question in questions], dim=1)
        # question_embeds = torch.cat([question_embeds, self.answer_embed(question_answers).transpose(0, 1)], dim=2)

        question_knowledges = inputs["question_knowledges"]
        logical_types = inputs["question_logical_types"]

        knowledge_embeds = self.knowledge_embed(question_knowledges).transpose(0, 1).mean(axis=-2)
        answer_embeds = self.answer_embed(question_answers).transpose(0, 1)
        logical_type_embeds = self.logical_type_embed(logical_types).transpose(0, 1)
        positional_embeds = self.positional_embed(input=interval_time).transpose(0, 1)
        positional_embeds_out = self.positional_embed_out(positions=interval_time).transpose(0, 1)

        # [seq_length, batch_size, embed_dim]
        question_feature_embeds = torch.cat([question_embeds,knowledge_embeds,logical_type_embeds], dim=-1) + positional_embeds
        key_padding_mask = torch.ones((question_feature_embeds.size(1), question_feature_embeds.size(0)), device=time_stamps.device).bool()
        for i in range(len(input_lengths)):
            key_padding_mask[i][:input_lengths[i]] = False
        question_feature_embeds= self.attention_layer_norm(question_feature_embeds)
        question_attention_embeds, attn_weights = self.multihead_attention(question_feature_embeds, question_feature_embeds, answer_embeds, key_padding_mask=key_padding_mask, attn_mask=attention_mask)
        # [seq_length, batch_size, embed_dim]
        question_attention_feature_embeds = torch.cat([question_embeds,knowledge_embeds,logical_type_embeds,question_attention_embeds], dim=-1)
        question_status_feature_embeds = torch.cat([question_embeds,knowledge_embeds,logical_type_embeds, answer_embeds.repeat(1, 1, self.config.multihead_attention_num_heads)], dim=-1)
        sequence_output = []
        for i in range(len(question_attention_feature_embeds)-1):
            hx_attention = self.bilstm(question_attention_feature_embeds[i+1], None if i==0 else hx_status)
            sequence_output.append(hx_attention)
            hx_status = self.bilstm(question_status_feature_embeds[i+1], None if i==0 else hx_status)

        sequence_output = torch.stack(sequence_output, dim=0)
        # embs = self.dropout(embs)
        # [batch_size, seq_length, embed_dim]
        out = self.out(sequence_output).transpose(0, 1)
        return out

    def loss(self, inputs, outputs, targets):
        # [batch_size embed_dim, seq_length]
        out = outputs.transpose(1, 2)
        question_answers = inputs["question_answers"][:, 1:]
        loss = self.loss_function(out, question_answers)
        return loss

    def evaluate(self, input_ids, input_lens, input_tags):
        features = self.forward(input_ids)
        loss = self.crf.calculate_loss(features, input_lens, input_tags)
        tags, confidences = self.crf.obtain_labels(features, input_lens)
        return tags, confidences, loss

    def predict(self, inputs):
        with torch.no_grad():
            self.eval()
            # [batch_size=1, seq_length]
            time_stamps = inputs["time_stamps"]
            # [batch_size=1, seq_length]
            question_answers = inputs["question_answers"]
            # [seq_length, batch_size=1, embed_dim]
            if 'questions_embeds' in inputs:
                question_embeds = inputs["questions_embeds"].transpose(0, 1)
            else:
                questions = inputs["questions"]
                question_embeds = torch.stack([self.question_embedding_model.encode(question, convert_to_tensor=True, show_progress_bar=False, device=time_stamps.device) for question in questions], dim=1)
            question_knowledges = inputs["question_knowledges"]
            logical_types = inputs["question_logical_types"]
            # history info
            feature_history = inputs.get("feature_history", {})
            gru_hidden_status = feature_history.get("gru_hidden_status", None)
            #last_time_stamps = feature_history.get("last_time_stamps", torch.tensor([0], device=time_stamps.device))
            #interval_time = torch.clamp(torch.tensor([[(time_stamps-last_time_stamps) // 300 + len(feature_history)]], device=time_stamps.device), min=0, max=31400)
            # [seq_length, batch_size=1, embed_dim]
            question_feature_embeds_history = feature_history.get("question_feature_embeds_history", [])
            # [seq_length, batch_size=1, embed_dim]
            answer_embeds_history = feature_history.get("answer_embeds_history", [])
            # [seq_length, batch_size=1, embed_dim]
            time_stamps_history = feature_history.get("time_stamps_history", torch.tensor([], device=time_stamps.device))

            knowledge_embeds = self.knowledge_embed(question_knowledges).transpose(0, 1).mean(axis=-2)
            answer_embeds = self.answer_embed(question_answers).transpose(0, 1)
            logical_type_embeds = self.logical_type_embed(logical_types).transpose(0, 1)
            last_interval_time, attention_mask = self.get_last_time_mask(torch.cat([time_stamps_history, time_stamps], dim=-1))
            positional_embeds = self.positional_embed(positions=torch.tensor([[len(question_feature_embeds_history)]], device=time_stamps.device)).transpose(0, 1)
            positional_embeds_out = self.positional_embed_out(positions=last_interval_time[:, -1:]).transpose(0, 1)
            question_feature_embeds = torch.cat([question_embeds,knowledge_embeds,logical_type_embeds], dim=-1) + positional_embeds
            question_feature_embeds = self.attention_layer_norm(question_feature_embeds)
            if len(question_feature_embeds_history) != 0:
                # [seq_length, batch_size=1, embed_dim]
                key_padding_mask = torch.zeros((question_feature_embeds_history.size(1), question_feature_embeds_history.size(0)), device=time_stamps.device).bool()
                question_attention_embeds, attn_weights = self.multihead_attention(question_feature_embeds, question_feature_embeds_history, answer_embeds_history, key_padding_mask=key_padding_mask, attn_mask=attention_mask)
            else:
                question_attention_embeds = torch.zeros((1, 1, self.config.answer_embedding_output_dim*self.config.multihead_attention_num_heads), device=time_stamps.device)
                # [seq_length, batch_size=1, embed_dim]
            question_attention_feature_embeds = torch.cat([question_embeds,knowledge_embeds,logical_type_embeds,question_attention_embeds], dim=-1)
            question_status_feature_embeds = torch.cat([question_embeds,knowledge_embeds,logical_type_embeds,answer_embeds.repeat(1, 1, self.config.multihead_attention_num_heads)], dim=-1)

            sequence_output = []
            hx_attention = self.bilstm(question_attention_feature_embeds[0], gru_hidden_status)
            sequence_output.append(hx_attention)
            hx_status = self.bilstm(question_status_feature_embeds[0], gru_hidden_status)
            sequence_output = torch.stack(sequence_output, dim=0)
            #embs = self.dropout(embs)
            # [batch_size=1, seq_length, embed_dim]
            out = self.out(sequence_output).transpose(0, 1)
            # update history
            if len(question_feature_embeds_history) != 0:
                question_feature_embeds_history = torch.cat([question_feature_embeds_history, question_feature_embeds], dim=0)
                answer_embeds_history = torch.cat([answer_embeds_history, answer_embeds], dim=0)
                time_stamps_history = torch.cat([time_stamps_history, time_stamps], dim=-1)
            else:
                question_feature_embeds_history = question_feature_embeds
                answer_embeds_history = answer_embeds
                time_stamps_history = time_stamps
            feature_history["gru_hidden_status"] = hx_status
            feature_history["time_stamps_history"] = time_stamps_history
            feature_history["question_feature_embeds_history"] = question_feature_embeds_history
            feature_history["answer_embeds_history"] = answer_embeds_history

            self.train()
            return out[:, -1:, :], feature_history


    def optimizer(self):
        return optim.Adam