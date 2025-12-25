from hyperopt.pyll.base import Raise
from ailib.models.base_model import BaseModel
from ailib.param.param import Param
from ailib.param import hyper_spaces
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from scipy.sparse import coo_matrix
from ailib.tools.utils_file import load_svmlight
import time
from pathlib import Path

class BaseMLModel():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._save_dir = Path("./outputs")/self.config.model_name/time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))

    def fit(self, X, y, *args, **kwargs):
        raise NotImplementedError

    def save(self, save_dir=None):
        raise NotImplementedError
