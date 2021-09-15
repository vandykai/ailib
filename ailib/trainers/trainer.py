"""Base Trainer."""

import typing
import time
import logging
from pathlib import Path
from collections import Iterable

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from ailib import tasks
from ailib.models.base_model import BaseModel
from ailib.metrics.base_metric import BaseMetric
from ailib.meters import AverageMeter, RecoderMeter
from ailib.tools.utils_time import Timer
from ailib.tools.utils_init import init_logger
from ailib.strategy import EarlyStopping
from ailib.tools.utils_statistic import grad_norm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

logger = logging.getLogger('__ailib__')

class Trainer:
    """
    ailib tranier.

    :param model: A :class:`BaseModel` instance.
    :param optimizer: A :class:`optim.Optimizer` instance.
    :param trainloader: A :class`Iterable` instance. The dataloader
        is used for training the model.
    :param validloader: A :class`Iterable` instance. The dataloader
        is used for validating the model.
    :param device: The desired device of returned tensor. Default:
        if None, use the current device. If `torch.device` or int,
        use device specified by user. If list, use data parallel.
    :param start_epoch: Int. Number of starting epoch.
    :param epochs: The maximum number of epochs for training.
        Defaults to 10.
    :param validate_interval: Int. Interval of validation.
    :param scheduler: LR scheduler used to adjust the learning rate
        based on the number of epochs.
    :param clip_norm: Max norm of the gradients to be clipped.
    :param patience: Number fo events to wait if no improvement and
        then stop the training.
    :param key: Key of metric to be compared.
    :param checkpoint: A checkpoint from which to continue training.
        If None, training starts from scratch. Defaults to None.
        Should be a file-like object (has to implement read, readline,
        tell, and seek), or a string containing a file name.
    :param save_dir: Directory to save trainer.
    :param save_all: Bool. If True, save `Trainer` instance; If False,
        only save model. Defaults to False.
    :param verbose: 0, 1, or 2. Verbosity mode. 0 = silent,
        1 = verbose, 2 = one log line per epoch.
    """

    def __init__(
        self,
        model: BaseModel,
        optimizer: optim.Optimizer,
        trainloader: Iterable,
        validloader: Iterable,
        device: typing.Union[torch.device, int, list, None] = None,
        start_epoch: int = 1,
        epochs: int = 10,
        validate_interval: typing.Optional[int] = None,
        scheduler: typing.Any = None,
        clip_norm: typing.Union[float, int] = None,
        patience: typing.Optional[int] = None,
        should_decrease: bool = False,
        key: typing.Any = None,
        checkpoint: typing.Union[str, Path] = None,
        save_dir: typing.Union[str, Path] = None,
        save_all: bool = False,
        verbose: int = 1,
        loss_proxy: callable = None,
        metric_proxy: callable = None,
        debug: bool = False,
        **kwargs
    ):
        """Base Trainer constructor."""
        self._load_model(model, device)
        self._load_dataloader(trainloader, validloader, validate_interval)

        self._optimizer = optimizer
        self._scheduler = scheduler
        self._clip_norm = clip_norm
        self._criterions = self._task.losses

        if not key:
            self._key = str(self._task.metrics[0])
        else:
            self._key = key
        self._early_stopping = EarlyStopping(patience=patience, should_decrease=should_decrease)

        self._start_epoch = start_epoch
        self._epoch = start_epoch
        self._epochs = epochs
        self._iteration = 0
        self._verbose = verbose
        self._save_all = save_all
        self._loss_proxy = loss_proxy
        self._metric_proxy = metric_proxy
        self._debug = debug
        self._last_result = None
        self._load_path(checkpoint, save_dir)
        self._info_meter = RecoderMeter()
        self._train_loss_meter = AverageMeter()
        self._valid_loss_meter = AverageMeter()
        self._grad_norm_meter = AverageMeter()

    def _load_dataloader(
        self,
        trainloader: Iterable,
        validloader: Iterable,
        validate_interval: typing.Optional[int] = None
    ):
        """
        Load trainloader and determine validate interval.

        :param trainloader: A :class`Iterable` instance. The dataloader
            is used to train the model.
        :param validloader: A :class`Iterable` instance. The dataloader
            is used to validate the model.
        :param validate_interval: int. Interval of validation.
        """
        self._trainloader = trainloader
        self._validloader = validloader
        if not validate_interval:
            self._validate_interval = len(self._trainloader)
        else:
            self._validate_interval = validate_interval

    def _load_model(
        self,
        model: BaseModel,
        device: typing.Union[torch.device, int, list, None] = None
    ):
        """
        Load model.

        :param model: :class:`BaseModel` instance.
        :param device: The desired device of returned tensor. Default:
            if None, use the current device. If `torch.device` or int,
            use device specified by user. If list, use data parallel.
        """
        # if not isinstance(model, BaseModel):
        #     raise ValueError(
        #         'model should be a `BaseModel` instance.'
        #         f' But got {type(model)}.'
        #     )

        self._task = model.config.task
        self._data_parallel = False
        self._model = model
        self._model_name = model.config.model_name
        if isinstance(device, list) and len(device):
            self._data_parallel = True
            self._device = device[0]
            self._model = torch.nn.DataParallel(self._model, device_ids=device, output_device=self._device)
        else:
            if not (isinstance(device, torch.device) or isinstance(device, int) or isinstance(device, str)):
                device = torch.device(
                    "cuda" if torch.cuda.is_available() else "cpu")
            self._device = device
        if isinstance(self._device, int):
            self._device = torch.device(f"cuda:{self._device}")
        self._model.to(self._device)

    def _load_path(
        self,
        checkpoint: typing.Union[str, Path],
        save_dir: typing.Union[str, Path],
    ):
        """
        Load save_dir and Restore from checkpoint.

        :param checkpoint: A checkpoint from which to continue training.
            If None, training starts from scratch. Defaults to None.
            Should be a file-like object (has to implement read, readline,
            tell, and seek), or a string containing a file name.
        :param save_dir: Directory to save trainer.

        """
        if not save_dir:
            save_dir = Path("./outputs")/self._model_name/time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))

        self._save_dir = Path(save_dir)
        # Restore from checkpoint

        if checkpoint:
            if self._save_all:
                self.restore(checkpoint)
            else:
                self.restore_model(checkpoint)

    def _backward(self, loss):
        """
        Computes the gradient of current `loss` graph leaves.

        :param loss: Tensor. Loss of model.

        """
        self._optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if self._clip_norm:
            nn.utils.clip_grad_norm_(
                self._model.parameters(), self._clip_norm
            )
        self._optimizer.step()

    def _run_scheduler(self, metrics, step, type=''):
        """Run scheduler."""
        if self._scheduler is not None:
            if type=='batch_step' and hasattr(self._scheduler, "batch_step"):
                self._scheduler.batch_step(metrics, step)
            elif type=='epoch_step' and hasattr(self._scheduler, "epoch_step"):
                self._scheduler.epoch_step(metrics, step)
            elif type=='epoch_step' and hasattr(self._scheduler, "step"):
                self._scheduler.step()
            if type=='epoch_step':
                lr = [item['lr'] for item in self._optimizer.state_dict()['param_groups']]
                logger.info(f"current lr: {lr}")

    def run(self):
        """
        Train model.

        The processes:
            Run each epoch -> Run scheduler -> Should stop early?

        """
        timer = Timer()
        for epoch in range(self._start_epoch, self._epochs + 1):
            self._run_epoch()
            self._epoch += 1
            self.save("trainer_latest.pt")
            if self._early_stopping.should_stop_early:
                break
        logger.info(f"best model path: {self._save_dir.joinpath('model.pt')}")
        if self._verbose:
            tqdm.write(f'cost time: {timer.time}s')

    # loss is (inputs, outputs, targets) while metric is (inputs, targets, outputs)
    def _caculate_loss(self, inputs, outputs, targets):
        # Caculate all losses and sum them up
        if self._loss_proxy:
            loss = torch.sum(
                *[self._loss_proxy(c, inputs, outputs, targets) for c in self._criterions]
            )
        elif hasattr(self._model, 'loss'):
            loss = self._model.loss(inputs, outputs, targets)
        else:
            loss = torch.sum(
                *[c(outputs, targets["target"]) for c in self._criterions]
            )
        return loss

    def _run_epoch(self):
        """
        Run each epoch.

        The training steps:
            - Get batch and feed them into model
            - Get outputs. Caculate all losses and sum them up
            - Loss backwards and optimizer steps
            - Evaluation
            - Update and output result

        """
        # Get total number of batch
        num_batch = len(self._trainloader)
        with tqdm(enumerate(self._trainloader), total=num_batch,
                  disable=not self._verbose) as pbar:
            self._model.train()
            for step, (inputs, targets) in pbar:
                outputs = self._model(inputs)
                # Caculate all losses and sum them up
                loss = self._caculate_loss(inputs, outputs, targets)
                self._backward(loss)

                self._model.attack() # 在embedding上添加对抗扰动
                outputs = self._model(inputs)
                loss_adv = self._caculate_loss(inputs, outputs, targets)
                self._backward(loss_adv)
                self._model.restore() # 恢复embedding参数

                self._info_meter.update("train_loss", loss.item())
                self._info_meter.update("grad_norm", grad_norm(self._model.parameters(), float('inf')))
                # batch lr scheduler
                self._run_scheduler(metrics=loss.item(), step=self._iteration, type='batch_step')
                # Set progress bar
                pbar.set_description(f'Epoch {self._epoch}/{self._epochs}')
                pbar.set_postfix(grad_norm=f'{self._info_meter.avg["grad_norm"]:.5f}', loss=f'{loss.item():.5f}')

                # Run validate
                self._iteration += 1
                if self._iteration % self._validate_interval == 0:
                    pbar.update(1)
                    # put create dir here to avoid terminal stop and make empty dir
                    if not Path(self._save_dir).exists():
                        Path(self._save_dir).mkdir(parents=True)
                        init_logger(self._save_dir.joinpath('train.log'))
                    if self._verbose:
                        logger.info({
                            "Epoch": f'{self._epoch}/{self._epochs}',
                            "Iter": self._iteration,
                            "GradNorm": f'{self._info_meter.avg["grad_norm"]:.3f}',
                            "Loss":f'{self._info_meter.avg["train_loss"]:.3f}'
                            })
                    self._last_result = self.evaluate(self._validloader)
                    if self._verbose:
                        logger.info({'validation':{k: v for k, v in self._last_result.items()}})
                    # Early stopping
                    self._early_stopping.update(self._last_result[self._key])
                    # epoch lr scheduler
                    self._run_scheduler(metrics=self._last_result[self._key], step=self._epoch, type='epoch_step')
                    if self._early_stopping.should_stop_early:
                        logger.info(f'Ran out of patience after {self._epoch} epoch. Stop training...')
                        break
                    elif self._early_stopping.is_best_so_far:
                        logger.info(f"Epoch {self._epoch}/{self._epochs}: best valid value improved to {self._early_stopping.best_so_far}")
                        self._save()
                    if self._debug:
                        self._info_meter.reset(keep_history=True)
                    else:
                        self._info_meter.reset(keep_history=False)

    def evaluate(self, dataloader: Iterable):
        """
        Evaluate the model.

        :param dataloader: A DataLoader object to iterate over the data.

        """
        result = dict()
        # Get total number of batch
        num_batch = len(dataloader)
        valid_loss_meter = AverageMeter()
        for metric in self._task.metrics:
            metric.reset()
        with torch.no_grad():
            self._model.eval()
            with tqdm(enumerate(dataloader), total=num_batch,
                  disable=not self._verbose) as pbar:
                for step, (inputs, targets) in pbar:
                    outputs = self._model(inputs)
                    loss = self._caculate_loss(inputs, outputs, targets)
                    valid_loss_meter.update(loss.item())
                    outputs = outputs.detach().cpu()
                    if self._metric_proxy:
                        for metric in self._task.metrics:
                            if isinstance(metric, BaseMetric):
                                metric.update(*self._metric_proxy(inputs, targets, outputs))
                    else:
                        for metric in self._task.metrics:
                            if isinstance(metric, BaseMetric):
                                metric.update(targets["target"].detach().cpu().numpy().tolist(), outputs.numpy().tolist())
            self._model.train()

        for metric in self._task.metrics:
            if isinstance(metric, BaseMetric):
                result[str(metric)] = metric.result()
        self._info_meter.update('valid_loss', valid_loss_meter.avg)
        result['loss'] = valid_loss_meter.avg
        return result

    def predicts(
        self,
        dataloader: Iterable
    ) -> np.array:
        """
        Generate output predictions for the input samples.

        :param dataloader: input DataLoader
        :return: predictions

        """
        # Get total number of batch
        num_batch = len(dataloader)
        with torch.no_grad():
            self._model.eval()
            predictions = []
            targets = []
            with tqdm(enumerate(dataloader), total=num_batch,
                disable=not self._verbose) as pbar:
                for step, (inputs, target) in pbar:
                    outputs = self._model(inputs).detach().cpu()
                    target = target["target"].detach().cpu()
                    predictions.append(outputs)
                    targets.append(target)
            self._model.train()
            return targets, predictions

    def predict(
        self,
        inputs
    ) -> torch.Tensor:
        """
        Generate output prediction for the input samples.

        :param dataloader: model inputs
        :return: prediction

        """
        with torch.no_grad():
            self._model.eval()
            outputs = self._model(inputs).detach().cpu()
            self._model.train()
            return outputs

    def _save(self):
        """Save."""
        if self._save_all:
            self.save("trainer.pt")
        else:
            self.save_model("model.pt")

    def save_model(self, file_name):
        """Save the model."""
        checkpoint = self._save_dir.joinpath(file_name)
        if self._data_parallel:
            model = self._model.module
        else:
            model = self._model
        state_dict = model.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].cpu()
        torch.save(state_dict, checkpoint)

    def save(self, file_name):
        """
        Save the trainer.

        `Trainer` parameters like epoch, best_so_far, model, optimizer
        and early_stopping will be savad to specific file path.

        :param path: Path to save trainer.

        """
        if not Path(self._save_dir).exists():
            Path(self._save_dir).mkdir(parents=True)
            init_logger(self._save_dir.joinpath('train.log'))
        checkpoint = self._save_dir.joinpath(file_name)
        if self._data_parallel:
            model = self._model.module
        else:
            model = self._model
        state_dict = model.state_dict()
        for key in state_dict:
            state_dict[key] = state_dict[key].cpu()
        state = {
            'epoch': self._epoch,
            'model': state_dict,
            'optimizer': self._optimizer.state_dict(),
            'early_stopping': self._early_stopping.state_dict(),
            'last_result': self._last_result
        }
        if self._scheduler:
            state['scheduler'] = self._scheduler.state_dict()
        torch.save(state, checkpoint)

    def restore_model(self, checkpoint: typing.Union[str, Path], strict=True):
        """
        Restore model.

        :param checkpoint: A checkpoint from which to continue training.

        """
        state = torch.load(checkpoint, map_location=self._device)
        if self._data_parallel:
            self._model.module.load_state_dict(state, strict)
        else:
            self._model.load_state_dict(state, strict)

    def restore(self, checkpoint: typing.Union[str, Path] = None, strict=True):
        """
        Restore trainer.

        :param checkpoint: A checkpoint from which to continue training.

        """
        state = torch.load(checkpoint, map_location=self._device)
        if self._data_parallel:
            self._model.module.load_state_dict(state['model'], strict)
        else:
            self._model.load_state_dict(state['model'], strict)
        self._optimizer.load_state_dict(state['optimizer'])
        self._start_epoch = state['epoch'] + 1
        self._epoch = state['epoch'] + 1
        self._early_stopping.load_state_dict(state['early_stopping'])
        self._last_result = state['last_result']
        if self._scheduler:
            self._scheduler.load_state_dict(state['scheduler'])

    def load_best_model(self, model_path="model.pt"):
        return self.restore_model(self._save_dir.joinpath(model_path))

    def plot_learning_curve(self, title='loss', ylim=(0, 5)):
        total_steps = len(self._info_meter.vals['train_loss'])
        x_1 = range(total_steps)
        x_2 = np.linspace(0, total_steps, len(self._info_meter.vals['valid_loss'])).round()
        figure(figsize=(6, 4))
        plt.plot(x_1, self._info_meter.vals['train_loss'], c='tab:red', label='train_loss')
        plt.plot(x_2, self._info_meter.vals['valid_loss'], c='tab:cyan', label='valid_loss')
        plt.ylim(*ylim)
        plt.xlabel('Training steps')
        plt.ylabel('loss')
        plt.title(f'{title} learning curve')
        plt.legend()
        plt.show()