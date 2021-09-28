from ailib import metrics
from enum import auto
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from functools import partial
from transformers import AutoTokenizer, AdamW
from ailib.datasets.competition.datafountain.product_comment_viewpoint_extraction.io import get_data, get_submit_example_df
from ailib.preprocessors.units import BatchPadding2D, BatchPadding3D
from ailib.models.seq_label.seq_label_lstm_crf import ModelParam, Model
from ailib.tasks import RegressionTask, ClassificationTask, ClassificationMultiLabelTask, RankingTask, NerTask
from ailib.trainers import Trainer
from ailib.tools.utils_statistic import regularization
from ailib.tools.utils_random import seed_everything
from ailib.torchtext.data import Field, ReversibleField, NestedField, LabelField, MultiLabelField, Example, Dataset as TDataset
from ailib.metrics.metrics_ner import SeqEntityScore, SpanEntityScore

seed_everything(123)

train_config = {
    'batch_size': 8,
    'device': 'cuda:2',
    'pretrained_model_path':'/home/wandikai/pretrain_model/HuggingFace/chinese-roberta-wwm-ext/hfl'
}

auto_tokenizer = AutoTokenizer.from_pretrained(train_config['pretrained_model_path'])
# tokenizer.add_tokens(exercise_preprocesor.context['latex_tokens']), len(exercise_preprocesor.context['latex_tokens'])
auto_tokenizer.add_tokens(['[SOS]', '[EOS]'], 2)
def tokenizer(text):
    text_list = list(text)
    return auto_tokenizer.convert_tokens_to_ids(text_list)
#tokenizer = partial(auto_tokenizer.encode, max_length=512, add_special_tokens=False, truncation=True)
#sos_id, eos_id, pad_id, unk_id = tokenizer('[SOS][EOS][PAD][UNK]')
pad_id, unk_id = auto_tokenizer.pad_token_id, auto_tokenizer.unk_token_id
text_field = Field(batch_first=True, tokenize=tokenizer, 
    init_token=None, eos_token=None, pad_token=pad_id, unk_token=unk_id, 
    fix_length=None, use_vocab=False, include_lengths=True)
index_field = Field(sequential=False, use_vocab=False)
label_field = Field(sequential=True, tokenize = lambda x:x, batch_first=True, init_token=None, eos_token=None, unk_token=None, include_lengths=True, is_target=True)

# document_field = NestedField(text_field, tokenize=None, fix_length=None, pad_token=0, include_lengths=True, use_vocab=False)
# doc_list = document_field.preprocess(content)
# (input_ids, doc_len, sentence_lens) = document_field.process([doc_list])

def get_dataset(mode, mydata):
    examples = []
    text_fields = [("text", text_field),("index", index_field), ("target", label_field)]
    for index, (text, bio_anno, label, bank_topic) in enumerate(mydata):
        examples.append(Example.fromlist([text, index, bio_anno], text_fields))
    return TDataset(examples, text_fields)

# token_pad = BatchPadding2D(init_token=None, pad_token=0, include_lengths=True)
# target_pad = BatchPadding2D(init_token=None, pad_token=0, include_lengths=True)
def train_pad_collate(samples):
    "Function that collect samples and adds padding. Flips token order if needed"
    input_ids = [item.text for item in samples]
    targets = [item.target for item in samples]
    input_ids, input_lengths =  text_field.process(input_ids, device=train_config['device'])
    targets, _ = label_field.process(targets, device=train_config['device'])
    attention_mask = (input_ids!=0)
    token_type_ids = torch.zeros_like(input_ids)
    return {"input_ids":input_ids, "input_lengths":input_lengths, "attention_mask":attention_mask,"token_type_ids":token_type_ids}, {"target":targets}

data_train, data_dev, data_test = get_data()
dataset_train, dataset_dev, dataset_test = get_dataset('train', data_train), get_dataset('dev', data_dev), get_dataset('test', data_test)
label_field.build_vocab(dataset_train)
train_loader = DataLoader(dataset_train, batch_size=train_config['batch_size'], shuffle=True, collate_fn=train_pad_collate)
valid_loader = DataLoader(dataset_dev, batch_size=train_config['batch_size'], shuffle=False, collate_fn=train_pad_collate)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=train_pad_collate)

model_param = ModelParam()
model_param['task'] = NerTask(num_classes = len(label_field.vocab) + 2, metrics=[SeqEntityScore(label_field.vocab.itos), SpanEntityScore(label_field.vocab.itos)])
model_param['embedding_input_dim'] = len(auto_tokenizer)
model_param['embedding_output_dim'] = 300
model_param['sos_tag_id'] = tokenizer('[SOS]')[0]
model_param['eos_tag_id'] = tokenizer('[EOS]')[0]
model_param['tag_vocab_size'] = len(label_field.vocab)
print(model_param)
model = Model(model_param.to_config()).to(train_config['device'])
#model.sentence_encoder.resize_token_embeddings(len(auto_tokenizer))
optimizer = AdamW(model.parameters(), lr=model_param['learning_rate'])
# a = next(iter(train_loader))[0]
# print(a['input_ids'].shape)
# print(model(a).shape)

# #model.sentence_encoder.resize_token_embeddings(len(tokenizer))
#scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                  #verbose=1, cooldown=0, min_lr=0, eps=1e-8)
#scheduler = linear_schedule_with_warmup(optimizer, num_warmup_steps=6, num_training_steps=20)
def metric_proxy(inputs, targets, outputs):
    y_preds = []
    y_trues = []
    for y_true, y_pred in zip(targets["target"].detach().cpu().numpy().tolist(), outputs):
        y_preds.append(y_pred)
        y_trues.append(y_true[:len(y_pred)])
    return y_trues, y_preds

trainer = Trainer(model=model,
                  optimizer = optimizer,
                  trainloader=train_loader,
                  validloader=valid_loader,
                  device=train_config['device'],
                  #scheduler=scheduler,
                  patience=5,
                  should_decrease = False,
                  metric_proxy = metric_proxy,
                  #validate_interval=10
                )
#trainer.run()
#trainer.restore_model('outputs/LSTM-CRF/2021-09-27-13-46-08/model.pt')
targets, predictions = trainer.predicts(test_loader)
targets = [ex for batch in targets for ex in batch]
predictions = [ex for batch in predictions for ex in batch]
predictions = [' '.join([label_field.vocab.itos[ind] for ind in ex]) for ex in predictions]


submit_example_df = get_submit_example_df('my_submit_example.csv')
submit_example_df['BIO_anno'] = predictions
submit_example_df.to_csv('my_submit_example.csv', index=False)