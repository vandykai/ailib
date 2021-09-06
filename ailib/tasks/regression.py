"""Regression task."""

from ailib.tasks.base_task import BaseTask
from ailib.metrics.metrics_regression import *

class RegressionTask(BaseTask):
    """Regression task.

    Examples:
        >>> regression_task = RegressionTask(num_classes=2)
        >>> regression_task.metrics = ['acc']
        >>> regression_task.output_shape
        (1,)
        >>> regression_task.output_dtype
        <class 'float'>
        >>> print(regression_task)
        Regression Task

    """

    TYPE = 'regression'

    def __init__(self, **kwargs):
        """Regression task."""
        super().__init__(**kwargs)
        self._num_classes = 1

    @property
    def num_classes(self) -> int:
        """:return: number of classes to classify."""
        return self._num_classes

    @classmethod
    def list_available_losses(cls) -> list:
        """:return: a list of available losses."""
        return ['mse']

    @classmethod
    def list_available_metrics(cls) -> list:
        """:return: a list of available metrics."""
        return ['mse']

    @property
    def output_shape(self) -> tuple:
        """:return: output shape of a single sample of the task."""
        return 1,

    @property
    def output_dtype(self):
        """:return: target data type, expect `float` as output."""
        return float

    def __str__(self):
        """:return: Task name as string."""
        return f'Regression Task'
