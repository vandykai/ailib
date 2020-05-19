from ailab.models.base_model import BaseModule
import torch, torch.nn.functional as F
from torch import ByteTensor, DoubleTensor, FloatTensor, HalfTensor, LongTensor, ShortTensor, Tensor
from torch import nn, optim, as_tensor
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, TensorDataset
from torch.nn.utils import weight_norm, spectral_norm

class DMN(BaseModule):
    def __init__(self, vocab_size, hidden_size=80, output_size=2, dropout_p=0.1):
        super(DMN, self).__init__()

        self.hidden_size = hidden_size
        self.embed = nn.Embedding(vocab_size, hidden_size) #sparse=True)
        self.input_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.question_gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        self.gate = nn.Sequential(
                            nn.Linear(hidden_size * 4, hidden_size),
                            nn.Tanh(),
                            nn.Linear(hidden_size, 1),
                            nn.Sigmoid()
                        )

        self.attention_grucell =  nn.GRUCell(hidden_size, hidden_size)
        self.memory_grucell = nn.GRUCell(hidden_size, hidden_size)
        self.answer_grucell = nn.GRUCell(hidden_size * 2, hidden_size)
        self.answer_fc = nn.Linear(hidden_size, output_size)

        self.dropout = nn.Dropout(dropout_p)

    def init_hidden(self, inputs):
        hidden = Variable(torch.zeros(1, inputs.size(0), self.hidden_size))
        return hidden.cuda() if USE_CUDA else hidden

    def init_weight(self):
        nn.init.xavier_uniform(self.embed.state_dict()['weight'])

        for name, param in self.input_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.question_gru.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.gate.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.attention_grucell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.memory_grucell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)
        for name, param in self.answer_grucell.state_dict().items():
            if 'weight' in name: nn.init.xavier_normal(param)

        nn.init.xavier_normal(self.answer_fc.state_dict()['weight'])
        self.answer_fc.bias.data.fill_(0)

    def forward(self, facts, fact_masks, questions, question_masks, num_decode, episodes=3, is_training=False):
        """
        facts : (B,T_C,T_I) / LongTensor in List # batch_size, num_of_facts, length_of_each_fact(padded)
        fact_masks : (B,T_C,T_I) / ByteTensor in List # batch_size, num_of_facts, length_of_each_fact(padded)
        questions : (B,T_Q) / LongTensor # batch_size, question_length
        question_masks : (B,T_Q) / ByteTensor # batch_size, question_length
        """
        # Input Module
        C = [] # encoded facts
        for fact, fact_mask in zip(facts, fact_masks):
            embeds = self.embed(fact)
            if is_training:
                embeds = self.dropout(embeds)
            hidden = self.init_hidden(fact)
            outputs, hidden = self.input_gru(embeds, hidden)
            real_hidden = []

            for i, o in enumerate(outputs): # B,T,D
                real_length = fact_mask[i].data.tolist().count(0) 
                real_hidden.append(o[real_length - 1])

            C.append(torch.cat(real_hidden).view(fact.size(0), -1).unsqueeze(0))

        encoded_facts = torch.cat(C) # B,T_C,D

        # Question Module
        embeds = self.embed(questions)
        if is_training:
            embeds = self.dropout(embeds)
        hidden = self.init_hidden(questions)
        outputs, hidden = self.question_gru(embeds, hidden)

        if isinstance(question_masks, torch.autograd.variable.Variable):
            real_question = []
            for i, o in enumerate(outputs): # B,T,D
                real_length = question_masks[i].data.tolist().count(0) 
                real_question.append(o[real_length - 1])
            encoded_question = torch.cat(real_question).view(questions.size(0), -1) # B,D
        else: # for inference mode
            encoded_question = hidden.squeeze(0) # B,D

        # Episodic Memory Module
        memory = encoded_question
        T_C = encoded_facts.size(1)
        B = encoded_facts.size(0)
        for i in range(episodes):
            hidden = self.init_hidden(encoded_facts.transpose(0, 1)[0]).squeeze(0) # B,D
            for t in range(T_C):
                #TODO: fact masking
                #TODO: gate function => softmax
                z = torch.cat([
                                    encoded_facts.transpose(0, 1)[t] * encoded_question, # B,D , element-wise product
                                    encoded_facts.transpose(0, 1)[t] * memory, # B,D , element-wise product
                                    torch.abs(encoded_facts.transpose(0,1)[t] - encoded_question), # B,D
                                    torch.abs(encoded_facts.transpose(0,1)[t] - memory) # B,D
                                ], 1)
                g_t = self.gate(z) # B,1 scalar
                hidden = g_t * self.attention_grucell(encoded_facts.transpose(0, 1)[t], hidden) + (1 - g_t) * hidden

            e = hidden
            memory = self.memory_grucell(e, memory)

        # Answer Module
        answer_hidden = memory
        start_decode = Variable(LongTensor([[word2index['<s>']] * memory.size(0)])).transpose(0, 1)
        y_t_1 = self.embed(start_decode).squeeze(1) # B,D

        decodes = []
        for t in range(num_decode):
            answer_hidden = self.answer_grucell(torch.cat([y_t_1, encoded_question], 1), answer_hidden)
            decodes.append(F.log_softmax(self.answer_fc(answer_hidden),1))
        return torch.cat(decodes, 1).view(B * num_decode, -1)