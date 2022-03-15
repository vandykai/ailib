import random
import typing
from collections import defaultdict
from functools import partial
from pathlib import Path
from random import choice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from ailib.local.ali.dataset_file import FileDataset
from ailib.local.ali.date import Date
from ailib.loss_function import SoftmaxMultiLabelLoss
from ailib.models.base_model import BaseModel
from ailib.models.base_model_param import BaseModelParam
from ailib.models.text_cls import Transformer, TransformerParam
from ailib.modules.bert_module import BertModule
from ailib.modules.multihead_attention import MultiheadAttention
from ailib.param import hyper_spaces
from ailib.param.param import Param
from ailib.param.param_table import ParamTable
from ailib.preprocessors.units import BatchPadding2D, BatchPadding3D
from ailib.tasks import (ClassificationMultiLabelTask, ClassificationTask,
                         RankingTask, RegressionTask)
from ailib.text.basic_data import ch_en_punctuation
from ailib.tools.utils_encryption import md5, sha256
from ailib.tools.utils_file import load_fold_data, load_svmlight, save_svmlight
from ailib.tools.utils_init import init_logger
from ailib.tools.utils_ipython import display_html, display_img, display_pd
from ailib.tools.utils_name_parse import parse_activation
from ailib.tools.utils_persistence import (load_dill, load_pickle, save_dill,
                                           save_pickle)
from ailib.tools.utils_random import seed_everything
from ailib.tools.utils_statistic import regularization
from ailib.tools.utils_visualization import plot_dict_bar, plot_dict_line, get_score_bin_statistic
from ailib.trainers import Trainer
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm.auto import tqdm
from transformers import AdamW, AutoModel, AutoTokenizer
from treelib import Tree
from ailib.tools.utils_markdown import df2markdown, list2markdown, label2markdown
