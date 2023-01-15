from ailib.ml.models.cls_xgb import Model as XGBModel, ModelParam
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
logger = logging.getLogger('__ailib__')
from ailib.tools.utils_file import load_svmlight, get_svmlight_dim
from ailib.tools.utils_random import seed_everything
from scipy.sparse.csr import csr_matrix

class Model(XGBModel):

    def fit(self, X, y, treatment, *args, **kwargs):
        logger.info(f'start fitting')
        if not isinstance(X, csr_matrix):
            logger.info(f'convert data format to svmlight')
            _, w = get_svmlight_dim(X)
            X, y = load_svmlight(zip(*[X, [f'{w+1}:{it}' for it in treatment]]), y, on_memory=False)
        config_args = {
            "sample_weight":self.config.sample_weight,
            "early_stopping_rounds":self.config.early_stopping_rounds,
            "eval_metric":self.config.eval_metric,
            "eval_set":self.config.eval_set,
            "verbose":True
        }
        config_args.update(kwargs)
        if isinstance(config_args.get('eval_set', None), float):
            train_index, test_index = train_test_split(range(len(y)), test_size=self.config.eval_set, random_state=self.config.seed, stratify=y)
            X, X_test, y, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            if config_args.get('sample_weight', None) is not None:
                config_args['sample_weight'] = config_args['sample_weight'][train_index]
            eval_set = [(X, y), (X_test, y_test)]
            config_args['eval_set'] = eval_set
        logger.info(f'fit args:{config_args}')
        self._model.fit(X, y, *args, **config_args)

    def fit_lazy(self, X, y, treatment, block_size=4096, *args, **kwargs):
        result = []
        for i in range(0, X.shape[0], block_size):
            self._model.setp_fit(X[i:i+block_size], y[i:i+block_size], treatment[i:i+block_size], *args, **kwargs)

    def step_fit(self, X, y, treatment, *args, **kwargs):
        logger.info(f'start step fitting')
        if not isinstance(X, csr_matrix):
            _, w = get_svmlight_dim(X)
            X, y = load_svmlight(zip(*[X, [f'{w+1}:{it}' for it in treatment]]), y, on_memory=False)
        config_args = {
            "sample_weight":self.config.sample_weight,
            "early_stopping_rounds":self.config.early_stopping_rounds,
            "eval_metric":self.config.eval_metric,
            "eval_set":self.config.eval_set,
            "verbose":True
        }
        config_args.update(kwargs)
        if isinstance(config_args.get('eval_set', None), float):
            train_index, test_index = train_test_split(range(len(y)), test_size=self.config.eval_set, random_state=self.config.seed, stratify=y)
            X, X_test, y, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
            eval_set = [(X, y), (X_test, y_test)]
            config_args['eval_set'] = eval_set
        if not hasattr(self._model, '_Booster'):
            self._model.fit(X, y, *args, **config_args)
        else:
            self._model.fit(X, y, *args, **config_args, xgb_model=self._model.get_booster())

    def predict_proba(self, X):
        """
        Perform uplift on samples in X.
        """
        _, w = get_svmlight_dim(X)
        X_mod_trmnt= load_svmlight(zip(*[X, [f'{w+1}:{1}' for it in range(len(X))]]), None, on_memory=False)
        X_mod_ctrl = load_svmlight(zip(*[X, [f'{w+1}:{0}' for it in range(len(X))]]), None, on_memory=False)

        self.trmnt_preds_ = self._model.predict_proba(X_mod_trmnt)[:, 1]
        self.ctrl_preds_ = self._model.predict_proba(X_mod_ctrl)[:, 1]

        uplift = self.trmnt_preds_ - self.ctrl_preds_
        return np.array([[1-it, it] for it in uplift])