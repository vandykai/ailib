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
from ailib.models.base_model import BaseModel
from ailib.models.base_model_param import BaseModelParam
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
from typing import List, Callable

logger = logging.getLogger('__ailib__')

class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=False, with_multi_layer_perceptron=False):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self['model_name'] = "XGBoost"
        self['learning_rate'] = 5e-2
        self.add(Param(name='n_estimators', value=300, desc="n_estimators"))
        self.add(Param(name='max_depth', value=3, desc="max depth dim"))
        self.add(Param(name='min_child_weight', value=4, desc="min child weight"))
        self.add(Param(name='gamma', value=0.6, desc="gamma"))
        self.add(Param(name='subsample', value=0.8, desc="subsample"))
        self.add(Param(name='colsample_bytree', value=0.8, desc="colsample bytree"))
        self.add(Param(name='reg_alpha', value=5e-05, desc="reg alpha"))
        self.add(Param(name='objective', value='binary:logistic', desc="objective"))
        self.add(Param(name='obj', value=None, desc="obj"))
        self.add(Param(name='nthread', value=20, desc="nthread"))
        self.add(Param(name='scale_pos_weight', value=1, desc="scale pos weight"))
        self.add(Param(name='seed', value=123, desc="seed"))

        self.add(Param(name='sample_weight', value=None, desc="sample weight"))
        self.add(Param(name='early_stopping_rounds', value=None, desc="sample weight"))
        self.add(Param(name='eval_metric', value=['error','logloss', 'auc'], desc="eval metric"))
        self.add(Param(name='eval_set', value=0.2, desc="eval_set set by hand or a float number between (0, 1) to split train_set"))
        self.add(Param(name='verbose', value=True, desc="verbose"))

class XGBIterator(xgb.DataIter):
    def __init__(self, svm_file_paths: List[str]):
        self._file_paths = svm_file_paths
        self._it = 0
        super().__init__(cache_prefix=os.path.join(".", "cache"))

  
    def next(self, input_data: Callable):
        if self._it == len(self._file_paths):
            # return 0 to let XGBoost know this is the end of iteration
            return 0
        X, y = load_svmlight_file(self._file_paths[self._it])
        input_data(
            data=X,
            label=y,
            weight=None,
        )
        self._it += 1
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1

    def reset(self):
        """Reset the iterator to its beginning"""
        self._it = 0

class Model():
    def __init__(self, config):
        super().__init__()
        self.config = config
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
            self.step_fit(X[i:i+block_size], y[i:i+block_size], *args, **kwargs)

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
        plt.title('score distribute')
        _ = plt.hist(y_pred[:, 1], bins=100)


