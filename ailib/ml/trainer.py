from ailib.ml.base_model import BaseMLModel
from ailib.param.param_table import ParamTable
from ailib.param.param import Param
import typing
import torch

class TrainParam(ParamTable):
    def __init__(self):
        super().__init__()
        self.add(Param(name='model', desc="Model."))
        self.add(Param(name='trainloader', desc="File path or Iterable data."))
        self.add(Param(name='validloader', desc="File path or Iterable data."))
        self.add(Param(name='device', desc="typing.Union[torch.device, int, list, None]."))
        self.add(Param(name='save_dir', desc="typing.Union[str, Path]."))
        self.add(Param(name='debug', desc="Whether debug."))

class Trainer:
    def __init__(self, config):
        super().__init__()
        self.config = config

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
    def run(self):

    def 


        
