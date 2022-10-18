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
logger = logging.getLogger('__ailib__')
from ailib.tools.utils_file import load_svmlight, get_svmlight_dim
from ailib.tools.utils_random import seed_everything
from scipy.sparse.csr import csr_matrix

class Model(XGBModel):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._model =  xgb.XGBClassifier(
            learning_rate=config.learning_rate,
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_child_weight=config.min_child_weight,
            gamma=config.gamma,
            subsample=config.subsample,
            colsample_bytree=config.colsample_bytree,
            reg_alpha=config.reg_alpha,
            objective=config.objective,
            obj=config.obj,
            nthread=config.nthread,
            scale_pos_weight=config.scale_pos_weight,
            use_label_encoder=False,
            seed=config.seed
        )
        self._save_dir = Path("./outputs")/config.model_name/time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))


    def fit(self, X, y, *args, **kwargs):
        logger.info(f'start fitting')
        if not isinstance(X, csr_matrix):
            logger.info(f'convert data format to svmlight')
            X, y = load_svmlight(X, y, on_memory=False)
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

    def fit_lazy(self, X, y, block_size=4096, *args, **kwargs):
        result = []
        for i in range(0, X.shape[0], block_size):
            self._model.setp_fit(X[i:i+block_size], y[i:i+block_size], *args, **kwargs)

    def step_fit(self, X, y, *args, **kwargs):
        logger.info(f'start step fitting')
        if not isinstance(X, csr_matrix):
            X, y = load_svmlight(X, y, on_memory=False)
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
    
    def save(self, file_name=None):
        self._save_dir.mkdir(parents=True)
        if file_name is not None:
            checkpoint = self._save_dir.joinpath(file_name)
        else:
            checkpoint = self._save_dir.joinpath("model.pt")
        self._model.save_model(checkpoint)
        logger.info(f'model saved at :{checkpoint}')

    def load(self, file_path=None):
        if file_path is None:
            file_path = self._save_dir.joinpath("model.pt")
        self._model.load_model(file_path)

    def predict(self, X):
        return self._model.predict(X)

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    def predict_proba_lazy(self, X, block_size=4096):
        result = []
        for i in range(0, X.shape[0], block_size):
            result.append(self._model.predict_proba(X[i:i+block_size]))
        return np.concatenate(result, axis=0)

    def get_feature_importantce_by_tree_df(self, fmap=None, sep='\t', detail=False):
        if type(fmap) in [str, pathlib.PosixPath]:
            fmap = pd.read_csv(fmap, sep=sep, names=['name', 'id'])
            fmap['id'] = fmap['id'].map(lambda x:f'f{x}')
            fmap = fmap.set_index('id').to_dict()['name']
        tree_df = self._model.get_booster().trees_to_dataframe().sort_values('Gain', ascending=False)

        if fmap:
            tree_df['Feature'] = tree_df['Feature'].map(lambda x:fmap.get(x, x))
        if not detail:
            tree_df = tree_df.groupby('Feature').agg(Gain=('Gain','mean'), Cover=('Cover','mean')).sort_values(by=['Gain', 'Cover'], ascending=False).reset_index()
        return tree_df
    
    def result_graph(self, X, y):
        y_pred = self.predict_proba(X)
        plot_cls_result(y, y_pred)



