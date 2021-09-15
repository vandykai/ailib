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
from .trainer import Trainer

logger = logging.getLogger('__ailib__')

class RankTrainer(Trainer):

    def evaluate(self, dataloader: Iterable):
        result = dict()
        y_true, y_pred, id_left = self.predicts(dataloader)
        for metric in self._task.metrics:
            if isinstance(metric, BaseMetric):
                result[str(metric)] = self._eval_metric_on_data_frame(metric, id_left, y_true, y_pred)
        return result

    @classmethod
    def _eval_metric_on_data_frame(
        cls,
        metric: BaseMetric,
        id_left: typing.Any,
        y_true: typing.Union[list, np.array],
        y_pred: typing.Union[list, np.array]
    ):
        """
        Eval metric on data frame.

        This function is used to eval metrics for `Ranking` task.

        :param metric: Metric for `Ranking` task.
        :param id_left: id of input left. Samples with same id_left should
            be grouped for evaluation.
        :param y_true: Labels of dataset.
        :param y_pred: Outputs of model.
        :return: Evaluation result.

        """
        eval_df = pd.DataFrame(data={
            'id': id_left,
            'true': y_true,
            'pred': y_pred
        })
        assert isinstance(metric, BaseMetric)
        val = eval_df.groupby(by='id').apply(
            lambda df: metric(df['true'].values, df['pred'].values)
        ).mean()
        return val

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
            id_left = []
            predictions = []
            targets = []
            with tqdm(enumerate(dataloader), total=num_batch,
                disable=not self._verbose) as pbar:
                for step, (inputs, target) in pbar:
                    outputs = self._model(inputs).detach().cpu().numpy().tolist()
                    target = target["target"].detach().cpu().numpy().tolist()
                    predictions.extend(outputs)
                    targets.extend(target)
                    id_left.extend(inputs['id_left'])
            self._model.train()
            return targets, predictions, id_left