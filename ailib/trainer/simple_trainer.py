import torch
import torch.nn as nn
import time
import logging
from transformers import AdamW
from pathlib import Path
from itertools import count
from ailib.loss_function.label_smooth import LabelSmoothingLoss
from ailib.tools.utils_progressbar import ProgressBar
from ailib.lr_scheduler import ReduceLROnPlateau
from ailib.meter.meter_average import AverageMeter
from ailib.meter.meter_recoder import RecoderMeter
from ailib.metrics.metrics_cls import ClassificationScore
from ailib.tools.utils_persistence import save_model, load_model

logger = logging.getLogger()

def evaluate(config, model, dev_iter, device):
    loss_func = nn.CrossEntropyLoss()
    pbar = ProgressBar(n_total=len(dev_iter), desc="Evaluating")
    metric = ClassificationScore(label_field.vocab.itos)
    eval_loss = AverageMeter()
    model.to(device)
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(dev_iter):
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

def train(config, model, train_iter, device, max_epoch=None, early_stop_epoch=10):
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3,
                                  verbose=1, epsilon=1e-4, cooldown=0, min_lr=0, eps=1e-8)
    loss_func = LabelSmoothingLoss(0.1, config.n_classes, device=device)

    output_dir = Path("./outputs")/config.model_name/time.strftime("%Y-%m-%d-%H-%M",time.localtime(time.time()))
    if not output_dir.exists():
        os.makedirs(output_dir)

    if max_epoch is None:
        epoch_iter = count(1)
    else:
        epoch_iter = range(1, 1+max_epoch)
    max_grad_norm = 5.0
    result = {"best_f1":0, "no_improve_epochs":0}
    for epoch in epoch_iter:
        pbar = ProgressBar(n_total=len(train_iter), desc='Training')
        train_loss = AverageMeter()
        model.train()
        assert model.training
        for step, batch in enumerate(train_iter):
            input_feature, input_label = batch
            if input_feature is tuple:
                out = model(*input_feature)
            else:
                out = model(input_feature)
            loss = loss_func(out,input_label )
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            pbar(step=step, info={'loss': loss.item()})
            train_loss.update(loss.item(), n=1)
        print(" ")
        train_log = {'loss': train_loss.avg}
        if 'cuda' in str(device):
            torch.cuda.empty_cache()
        eval_log, class_info  = evaluate(config, model, dev_iter, device)
        sys.stdout.flush()
        logs = dict(train_log, **eval_log)
        info = f'Epoch: {epoch}/{max_epoch} - ' + "-".join([f' {key}: {value:.4f} ' for key, value in logs.items()])
        logger.info(info)
        scheduler.epoch_step(logs['eval_f1'], epoch)
        if logs['eval_f1'] > result["best_f1"]:
            result['no_improve_epochs'] = 0
            logger.info(f"Epoch {epoch}: eval_f1 improved from {result['best_f1']} to {logs['eval_f1']}")
            logger.info("save model to disk.")
            result['best_f1'] = logs['eval_f1']
            best_model_path = output_dir/'best-model.bin'
            save_model(model, best_model_path, epoch=epoch, model_name=config.model_name, logs=logs)
            logger.info("Eval Score:")
            for key, value in class_info.items():
                info = f"Subject: {key} - Acc: {value['acc']} - Recall: {value['recall']} - F1: {value['f1']}"
                logger.info(info)
        else:
            result['no_improve_epochs'] += 1
        if result['no_improve_epochs'] > early_stop_epoch:
            break
    last_model_path = output_dir/'last-model.bin'
    save_model(model, last_model_path, epoch=epoch, model_name = config.model_name, logs=logs)
    result["best_model_path"] = model_path
    return result