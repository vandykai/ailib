"""Base callback."""
import abc

import numpy as np

from ailib.data_pack import DataPack


class BaseCallback(abc.ABC):
    """
    DataGenerator callback base class.

    To build your own callbacks, inherit `BaseCallback`
    and overrides corresponding methods.

    A batch is processed in the following way:

    - slice data pack based on batch index
    - handle `on_batch_data_pack` callbacks
    - unpack data pack into x, y
    - handle `on_batch_x_y` callbacks
    - return x, y

    """

    def on_batch_data_pack(self, data_pack: DataPack):
        """
        `on_batch_data_pack`.

        :param data_pack: a sliced DataPack before unpacking.
        """

    @abc.abstractmethod
    def on_batch_unpacked(self, x: dict, y: np.ndarray):
        """
        `on_batch_unpacked`.

        :param x: unpacked x.
        :param y: unpacked y.
        """
