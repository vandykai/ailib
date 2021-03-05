import torch
import torch.nn as nn
import time
import sys
import logging
from transformers import AdamW
from pathlib import Path
from itertools import count
from ailib.loss_function.cls_label_smooth import LabelSmoothingLoss
from ailib.tools.utils_progressbar import ProgressBar
from ailib.lr_scheduler import ReduceLROnPlateau
from ailib.meters.meter_average import AverageMeter
from ailib.meters.meter_recoder import RecoderMeter
from ailib.metrics.metrics_cls import ClassificationScore
from ailib.tools.utils_persistence import save_model, load_model, save_dill

logger = logging.getLogger()

def evaluate(train_manager, dev_iter, input_handler=None):
    train_config = train_manager.config
    label_vocab = train_config.target_field.vocab.itos
    loss_func = train_config.loss_func
    pbar = ProgressBar(n_total=len(dev_iter), desc="Evaluating")
    metric = ClassificationScore(label_vocab)
    eval_loss = AverageMeter()
    model = train_config.model
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dev_iter):
            input_feature, input_label = batch
            if input_handler:
                predict_label, loss = input_handler(batch, model, loss_func)
            else:
                input_feature, input_label = batch
                if input_feature is tuple:
                    out = model(*input_feature)
                else:
                    out = model(input_feature)
                predict_label = torch.argmax(out, dim=1)
                loss = loss_func(out, input_label)
            metric.update(input_label, predict_label.cpu().numpy())
            eval_loss.update(val=loss.item(), n=predict_label.size(0))
            pbar(step=step)
    print(" ")
    eval_info, class_info = metric.result()
    eval_info = {f'eval_{key}': value for key, value in eval_info.items()}
    result = {'eval_loss': eval_loss.avg}
    result = dict(result, **eval_info)
    return result, class_info

def train(train_manager, train_iter, dev_iter, input_handler=None):
    train_config = train_manager.config
    model_config = train_config.model_config
    model = train_manager.config.model
    optimizer = train_config.optimizer
    scheduler = train_config.scheduler
    loss_func = train_config.loss_func
    output_dir = train_config.output_dir
    max_grad_norm = train_config.max_grad_norm
    label_vocab = train_config.target_field.vocab.itos
    epoch_iter = train_manager.epoch_iter
    train_manager.state.stop_train = False
    for epoch in epoch_iter:
        pbar = ProgressBar(n_total=len(train_iter), desc='Training')
        train_loss = AverageMeter()
        model.train()
        assert model.training
        for step, batch in enumerate(train_iter):
            if input_handler:
                predict_label, loss = input_handler(batch, model, loss_func)
            else:
                input_feature, input_label = batch
                if input_feature is tuple:
                    out = model(*input_feature)
                else:
                    out = model(input_feature)
                loss = loss_func(out, input_label)
            loss.backward()
            if train_config.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pbar(step=step, info={'loss': loss.item()})
            train_loss.update(loss.item(), n=1)
        print(" ")
        train_log = {'loss': train_loss.avg}
        if 'cuda' in str(train_config.device):
            torch.cuda.empty_cache()
        eval_log, class_info  = evaluate(train_manager, dev_iter, input_handler)
        sys.stdout.flush()
        logs = dict(train_log, **eval_log)
        info = f'Epoch: {epoch}/{train_config.max_epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(info)
        scheduler.epoch_step(logs['eval_f1'], epoch)
        train_manager.epoch_step(logs['loss'], logs['eval_f1'], epoch, model_name=model_config.model_name, logs=logs)
        if train_manager.state.score_improved:
            logger.info("Eval Score:")
            for key, value in class_info.items():
                info = f"Subject: {key} - Acc: {value['acc']} - Recall: {value['recall']} - F1: {value['f1']}"
                logger.info(info)
        if train_manager.state.stop_train:
            break