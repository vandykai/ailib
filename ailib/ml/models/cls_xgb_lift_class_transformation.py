from ailib.ml.models.cls_xgb import Model as XGBModel
import logging
import logging
import os
import pathlib
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from random import random

import dask.dataframe as dd
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
import xgboost as xgb
from ailib.param import hyper_spaces
from ailib.param.param import Param
from ailib.tools.utils_file import load_svmlight
from ailib.tools.utils_random import seed_everything
from ailib.tools.utils_visualization import plot_cls_result
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from matplotlib import pyplot as plt
from scipy import stats
from scipy.sparse.csr import csr_matrix
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import (auc, average_precision_score,
                             classification_report, precision_recall_curve,
                             roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm.auto import tqdm
from xgboost import plot_importance, plot_tree, to_graphviz
from ailib.tools.utils_file import load_svmlight, get_svmlight_dim
from ailib.tools.utils_random import seed_everything
from scipy.sparse.csr import csr_matrix

logger = logging.getLogger('__ailib__')

class Model(XGBModel):

    def fit(self, X, y, treatment, *args, **kwargs):
        y_mod = (np.array(y) == np.array(treatment)).astype(int)
        super().fit(X, y_mod, *args, **kwargs)

    def fit_lazy(self, X, y, treatment, block_size=4096, *args, **kwargs):
        result = []
        for i in range(0, X.shape[0], block_size):
            self._model.setp_fit(X[i:i+block_size], y[i:i+block_size], treatment[i:i+block_size], *args, **kwargs)

    def step_fit(self, X, y, treatment, *args, **kwargs):
        y_mod = (np.array(y) == np.array(treatment)).astype(int)
        super().step_fit(X, y_mod, *args, **kwargs)

    def predict_proba(self, X):
        """
        Perform uplift on samples in X.
        """
        uplift = 2 * self._model.predict_proba(X)[:, 1] - 1
        return np.array([[1-it, it] for it in uplift])