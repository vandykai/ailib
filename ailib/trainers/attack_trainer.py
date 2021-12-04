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
from ailib.tools.utils_statistic import grad_norm
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from .trainer import Trainer

logger = logging.getLogger('__ailib__')

class Trainer(Trainer):

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
                        init_logger(log_file=self._save_dir.joinpath('train.log'))
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
