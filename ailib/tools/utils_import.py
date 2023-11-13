import json
import logging
import math
import os
import pathlib
import random
import sys
import time
import typing
from collections import Counter, OrderedDict, defaultdict
from functools import partial
from pathlib import Path
from random import choice

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import xgboost as xgb
from sklearn.datasets import load_svmlight_file, load_svmlight_files
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm.auto import tqdm
from transformers import AdamW, AutoModel, AutoTokenizer
from treelib import Tree

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
from ailib.tools.utils_adversarial import RandomPerturbation
from ailib.tools.utils_check import check_df_label, clean_df_label, ismd5
from ailib.tools.utils_dict import dict_shrink, get_df_dict
from ailib.tools.utils_encryption import md5, sha256, to_base64
from ailib.tools.utils_feature import IV, get_sparse_feature_IV
from ailib.tools.utils_file import (count_line, get_files, get_files_size,
                                    get_svmlight_dim, load_files,
                                    load_fold_data, load_fold_data_iter,
                                    load_json, load_svmlight, read_lines,
                                    save_svmlight, save_to_file)
from ailib.tools.utils_init import init_logger
from ailib.tools.utils_ipython import display_html, display_img, display_pd
from ailib.tools.utils_markdown import (df2markdown, label2markdown,
                                        list2markdown, markdown2pdf)
from ailib.tools.utils_name_parse import parse_activation
from ailib.tools.utils_oss import (get_oss_download_url, get_oss_download_urls,
                                   get_oss_files, get_oss_files_size,
                                   get_oss_open_files, get_oss_upload_urls,
                                   load_oss_files, load_oss_fold_data,
                                   load_oss_fold_data_dict, open_oss_file,
                                   upload_file_to_oss, upload_fold_to_oss)
from ailib.tools.utils_persistence import (load_dill, load_model,
                                           load_model2oss, load_pickle,
                                           save_dill, save_model,
                                           save_model2oss, save_pickle)
from ailib.tools.utils_random import df_cut, df_cut_sample, seed_everything
from ailib.tools.utils_report import save_classification_report
from ailib.tools.utils_statistic import (get_distribute_dict,
                                         get_sample_rate_for_equal_dist,
                                         regularization)
from ailib.tools.utils_stream import MultiStreamReader
from ailib.tools.utils_url import url_parse
from ailib.tools.utils_visualization import (get_score_bin_statistic,
                                             plot_cls_auc, plot_cls_result,
                                             plot_dict_bar, plot_dict_bars,
                                             plot_dict_line,
                                             plot_time_distribute)
from ailib.trainers import Trainer

logger = logging.getLogger('__ailib__')
