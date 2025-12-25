"""Ner task."""

from ailib.tasks.base_task import BaseTask
from ailib.metrics.metrics_ner import *

class NerTask(BaseTask):
    """Ner task.

    Examples:
        >>> ner_task = NerTask(num_classes=2)
        >>> ner_task.metrics = ['ner_seq_score, ner_span_score']
        >>> ner_task.num_classes
        2
        >>> ner_task.output_shape
        (2,)
        >>> ner_task.output_dtype
        <class 'int'>
        >>> print(ner_task)
        Ner Task with 2 classes

    """

    TYPE = 'ner'

    def __init__(self, num_classes: int = 2, **kwargs):
        """Ner task."""
        super().__init__(**kwargs)
        if not isinstance(num_classes, int):
            raise TypeError("Number of classes must be an integer.")
        if num_classes < 2:
            raise ValueError("Number of classes can't be smaller than 2")
        self._num_classes = num_classes

    @property
    def num_classes(self) -> int:
        """:return: number of classes to classify."""
        return self._num_classes

    @classmethod
    def list_available_losses(cls) -> list:
        """:return: a list of available losses."""
        return ['cross_entropy']

    @classmethod
    def list_available_metrics(cls) -> list:
        """:return: a list of available metrics."""
        return ['ner_seq_score', 'ner_span_score']

    @property
    def output_shape(self) -> tuple:
        """:return: output shape of a single sample of the task."""
        return self._num_classes,

    @property
    def output_dtype(self):
        """:return: target data type, expect `int` as output."""
        return int

    def __str__(self):
        """:return: Task name as string."""
        return f'Ner Task with {self._num_classes} classes'
