import abc
import typing

from .unit import Unit


class StatefulUnit(Unit, metaclass=abc.ABCMeta):
    """
    Unit with inner state.

    Usually need to be fit before transforming. All information gathered in the
    fit phrase will be stored into its `context`.
    """

    def __init__(self):
        """Initialization."""
        self._context = {}

    @property
    def context(self):
        """Get current context."""
        return self._context

    @abc.abstractmethod
    def fit(self, input_: typing.Any):
        """Abstract base method, need to be implemented in subclass."""
