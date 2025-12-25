import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from functools import partial
from transformers import AutoTokenizer, AdamW
from ailib.datasets.competition.datafountain.product_comment_viewpoint_extraction.io import get_data, get_submit_example_df
from ailib.preprocessors.units import BatchPadding2D, BatchPadding3D
from ailib.models.text_cls import Transformer, TransformerParam
from ailib.tasks import RegressionTask, ClassificationTask, ClassificationMultiLabelTask, RankingTask
from ailib.trainers import Trainer
from ailib.tools.utils_statistic import regularization
from ailib.tools.utils_random import seed_everything

seed_everything(123)

train_config = {
    'batch_size': 8,
    'device': 'cpu',
    'pretrained_model_path':'/home/wandikai/pretrain_model/HuggingFace/chinese-roberta-wwm-ext/hfl'
}

class VDataset(Dataset):
    def __init__(self, mode='train', examples=None):
        self.mode = mode
        self.examples = examples

    def __getitem__(self, index):
        text, bio_anno, emotion, bank_topic = self.examples[index]
        return {"input_id":text, "target":emotion}

    def __len__(self):
        return len(self.examples)

tokenizer = AutoTokenizer.from_pretrained(train_config['pretrained_model_path'])
#tokenizer.add_tokens(exercise_preprocesor.context['latex_tokens']), len(exercise_preprocesor.context['latex_tokens'])
tokenizer = partial(tokenizer.encode, max_length=512, truncation=True)
token_pad = BatchPadding2D(init_token=None, pad_token=0, include_lengths=True)
def train_pad_collate(samples):
    "Function that collect samples and adds padding. Flips token order if needed"
    input_ids = [tokenizer(item['input_id']) for item in samples]
    targets = [item['target'] for item in samples]
    input_ids, input_lengths =  token_pad.transform(input_ids)
    input_ids = torch.tensor(input_ids, device=train_config['device'])
    targets = torch.tensor(targets, device=train_config['device'])
    attention_mask = (input_ids!=0)
    token_type_ids = torch.zeros_like(input_ids)
    return {"input_ids":input_ids, "attention_mask":attention_mask,"token_type_ids":token_type_ids}, {"target":targets}

data_train, data_dev, data_test = get_data()
dataset_train, dataset_dev, dataset_test = VDataset('train', data_train), VDataset('dev', data_dev), VDataset('test', data_test)

train_loader = DataLoader(dataset_train, batch_size=train_config['batch_size'], shuffle=True, collate_fn=train_pad_collate)
valid_loader = DataLoader(dataset_dev, batch_size=train_config['batch_size'], shuffle=False, collate_fn=train_pad_collate)
test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, collate_fn=train_pad_collate)

model_param = TransformerParam()
1/0.242, 1/0.138, 1/0.962
model_param['task'] = ClassificationTask(num_classes = 3, losses = [nn.CrossEntropyLoss(weight=torch.tensor([3., 3., 1.], device=train_config['device']))], metrics = ['kappa', 'acc'])
model_param['pretrained_model_out_dim'] = 768
model_param['pretrained_model_path'] = train_config['pretrained_model_path']
model = Transformer(model_param.to_config()).to(train_config['device'])
#model.sentence_encoder.resize_token_embeddings(len(tokenizer))
optimizer = AdamW(model.parameters(), lr=model_param['learning_rate'])
#scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                  #verbose=1, cooldown=0, min_lr=0, eps=1e-8)
#scheduler = linear_schedule_with_warmup(optimizer, num_warmup_steps=6, num_training_steps=20)
trainer = Trainer(model=model,
                  optimizer = optimizer,
                  trainloader=train_loader,
                  validloader=valid_loader,
                  device=train_config['device'],
                  #scheduler=scheduler,
                  patience=5,
                  should_decrease = False,
                  #metric_proxy = metric_proxy,
                  #validate_interval=10
                )
trainer.run()
trainer.load_best_model()
targets, predictions = trainer.predicts(test_loader)
predictions = torch.cat(predictions).argmax(dim=-1).numpy()

submit_example_df = get_submit_example_df()
submit_example_df['class'] = predictions
submit_example_df.to_csv('my_submit_example.csv', index=False)