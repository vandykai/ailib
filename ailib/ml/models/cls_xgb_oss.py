import logging
import os
import pathlib
import pickle
import time
import traceback
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from random import random
from typing import Callable, List

import dask.dataframe as dd
import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import scipy.sparse as sp
import seaborn as sns
import xgboost as xgb
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from matplotlib import pyplot as plt
from scipy import stats
from scipy.sparse.csr import csr_matrix
from sklearn import metrics
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import (auc, average_precision_score,
                             classification_report, confusion_matrix,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm.auto import tqdm
from xgboost import plot_importance, plot_tree, to_graphviz

from ailib.models.base_model import BaseModel
from ailib.models.base_model_param import BaseModelParam
from ailib.param import hyper_spaces
from ailib.param.param import Param
from ailib.tools.utils_encryption import to_base64
from ailib.tools.utils_file import load_svmlight
from ailib.tools.utils_markdown import df2markdown, markdown2pdf
from ailib.tools.utils_oss import get_oss_files_size, open_oss_file
from ailib.tools.utils_random import seed_everything
from ailib.tools.utils_stream import MultiStreamReader
from ailib.tools.utils_visualization import (get_score_bin_statistic,
                                             plot_cls_auc, plot_cls_result,
                                             plot_confusion_matrix,
                                             plot_fpr_tpr_curve, plot_ks_curve,
                                             plot_precision_recall_curve,
                                             plot_roc_curve,
                                             precision_recall_curve)


def is_memory_enough(file_size):
    mem = psutil.virtual_memory()
    logger.info(f'memory available:{mem.available} file_size:{file_size}')
    if mem.available > 2*file_size:
        return True
    return False

logger = logging.getLogger('__ailib__')

class ModelParam(BaseModelParam):

    def __init__(self, with_embedding=False, with_multi_layer_perceptron=False):
        super().__init__(with_embedding, with_multi_layer_perceptron)
        self['model_name'] = "XGBoost"
        self['learning_rate'] = 5e-2
        self.add(Param(name='tree_method', value='auto', desc="tree method"))
        self.add(Param(name='n_estimators', value=300, desc="nnumber of estimators"))
        self.add(Param(name='max_depth', value=3, desc="max depth"))
        self.add(Param(name='min_child_weight', value=4, desc="min child weight"))
        self.add(Param(name='gamma', value=0.6, desc="gamma"))
        self.add(Param(name='subsample', value=0.8, desc="subsample"))
        self.add(Param(name='colsample_bytree', value=0.8, desc="colsample bytree"))
        self.add(Param(name='reg_lambda', value=1, desc="reg_lambda"))
        self.add(Param(name='reg_alpha', value=5e-05, desc="reg alpha"))
        self.add(Param(name='objective', value='binary:logistic', desc="objective"))
        self.add(Param(name='obj', value=None, desc="obj"))
        self.add(Param(name='nthread', value=20, desc="nthread"))
        self.add(Param(name='scale_pos_weight', value=1, desc="scale pos weight"))
        self.add(Param(name='seed', value=123, desc="seed"))

        self.add(Param(name='sample_weight', value=None, desc="sample weight"))
        self.add(Param(name='early_stopping_rounds', value=None, desc="early stopping rounds"))
        self.add(Param(name='eval_metric', value=['auc', 'error','logloss'], desc="eval metric"))
        self.add(Param(name='eval_set', value=0.2, desc="eval_set set by hand or a float number between (0, 1) or a callable function to split train_set"))
        self.add(Param(name='verbose', value=True, desc="verbose"))
        self.add(Param(name='output_dir', value='outputs', desc="outputs"))

class XGBIterator(xgb.DataIter):
    def __init__(self, svm_file_paths: List[str], file_path_handler, model_name=None):
        self._file_paths = svm_file_paths
        self._file_path_handler = file_path_handler if file_path_handler is not None else lambda x:x
        self._it = 0
        self._pbar = tqdm(total=len(svm_file_paths))
        self._pbar.set_description(model_name)
        super().__init__(cache_prefix=os.path.join(".", f"cache-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')}"))

    def next(self, input_data: Callable):
        if self._it == len(self._file_paths):
            # return 0 to let XGBoost know this is the end of iteration
            return 0
        try:
            if type(self._file_paths[self._it]) == str and self._file_paths[self._it].startswith('oss:/'):
                X, y = load_svmlight_file(self._file_path_handler(self._file_paths[self._it]), zero_based=True)
            else:
                X, y = load_svmlight_file(self._file_path_handler(self._file_paths[self._it]), zero_based=True)
            if hasattr(self._file_paths[self._it], 'close'):
                self._file_paths[self._it].close()
            input_data(
                data=X,
                label=y,
                weight=None,
            )
        except Exception as e:
            logger.error(e)
        self._it += 1
        self._pbar.update(1)
        # Return 1 to let XGBoost know we haven't seen all the files yet.
        return 1

    def reset(self):
        """Reset the iterator to its beginning"""
        self._it = 0

class Model():
    def __init__(self, config):
        super().__init__()
        self.config = config
        self._save_dir = Path(f"./{config.output_dir}")/config.model_name/time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))

    def reset(self):
        self.train_Xy = None
        self.test_Xy = None
        self.oot_Xy = None
        self.evals_result = {}

    def fit(self, svm_file_paths, file_path_handler=None, **kwargs):
        self.reset()
        svm_file_paths = np.array(svm_file_paths)
        file_content_length = get_oss_files_size(svm_file_paths)
        logger.info(f'start fitting')
        logger.info(f'model param: {self.config}')
        config_args = {
            "nthread":self.config.nthread,
            'objective':self.config.objective,
            "eta":self.config.learning_rate,
            "gamma":self.config.gamma,
            "max_depth":self.config.max_depth,
            "min_child_weight":self.config.min_child_weight,
            "subsample":self.config.subsample,
            "colsample_bytree":self.config.colsample_bytree,
            "reg_lambda":self.config.reg_lambda,
            "reg_alpha":self.config.reg_alpha,
            "early_stopping_rounds":self.config.early_stopping_rounds,
            "eval_metric":self.config.eval_metric,
            "tree_method":self.config.tree_method,
            "scale_pos_weight":self.config.scale_pos_weight
        }
        if self.config.eval_set is not None:
            if isinstance(self.config.eval_set, float):
                train_index, test_index = train_test_split(range(len(svm_file_paths)), test_size=self.config.eval_set, random_state=self.config.seed)
                train_svm_files, test_svm_files = svm_file_paths[train_index], svm_file_paths[test_index]
            elif callable(self.config.eval_set):
                train_svm_files, test_svm_files = self.config.eval_set(svm_file_paths)
            else:
                raise ValueError(f'eval_set:{self.config.eval_set} must be float or callable or None')
            if is_memory_enough(file_content_length):
                logger.info(f'memory enough, load svmfile into memory')
                self.train_Xy = xgb.DMatrix(*load_svmlight_file(MultiStreamReader(map(file_path_handler, train_svm_files)), zero_based=True))
                self.test_Xy = xgb.DMatrix(*load_svmlight_file(MultiStreamReader(map(file_path_handler, test_svm_files)), zero_based=True))
            else:
                logger.info(f'memory not enough, load svmfile into externel memory')
                self.train_Xy = xgb.DMatrix(XGBIterator(train_svm_files, file_path_handler, self.config.model_name))
                self.test_Xy = xgb.DMatrix(XGBIterator(test_svm_files, file_path_handler, self.config.model_name))
            if self.config.sample_weight is not None:
                self.train_Xy.set_weight(self.config.sample_weight[train_index])
            eval_set = [(self.train_Xy, "train"),(self.test_Xy, "test")]
        else:
            if is_memory_enough(file_content_length):
                logger.info(f'memory enough, load svmfile into memory')
                self.train_Xy = xgb.DMatrix(*load_svmlight_file(MultiStreamReader(map(file_path_handler, svm_file_paths)), zero_based=True))
            else:
                logger.info(f'memory not enough, load svmfile into externel memory')
                self.train_Xy = xgb.DMatrix(XGBIterator(svm_file_paths, file_path_handler, self.config.model_name))
            eval_set = self.config.eval_set
        logger.info(f'fit args:{config_args}')
        self._model = xgb.train(params=config_args, dtrain=self.train_Xy, num_boost_round=self.config.n_estimators,
            evals=eval_set, obj=self.config.obj, early_stopping_rounds = self.config.early_stopping_rounds,
            evals_result=self.evals_result, verbose_eval=self.config.verbose
        )
        logger.info(f'evals_result:\n{self.evals_result}')

    def save_model(self, file_name=None):
        self._save_dir.mkdir(parents=True, exist_ok=True)
        if file_name is not None:
            checkpoint = self._save_dir.joinpath(file_name)
        else:
            checkpoint = self._save_dir.joinpath("model.pt")
        self._model.save_model(checkpoint)
        logger.info(f'model saved at :{checkpoint}')

    def load_model(self, file_path=None):
        if file_path is None:
            file_path = self._save_dir.joinpath("model.pt")
        self._model.load_model(file_path)

    def predict(self, X):
        y_predict = self._model.predict(X)
        return np.array([1 if it >= 0.5 else 0 for it in y_predict])

    def predict_proba(self, X):
        y_predict = self._model.predict(X)
        return np.array([[1-it, it] for it in y_predict])

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
        elif type(fmap) in [pd.DataFrame]:
            fmap['id'] = fmap['id'].map(lambda x:f'f{x}')
            fmap = fmap.set_index('id').to_dict()['name']
        tree_df = self._model.trees_to_dataframe().sort_values('Gain', ascending=False)

        if fmap:
            tree_df['Feature'] = tree_df['Feature'].map(lambda x:fmap.get(x, x))
        if not detail:
            tree_df = tree_df.groupby('Feature').agg(Gain=('Gain','mean'), Cover=('Cover','mean')).sort_values(by=['Gain', 'Cover'], ascending=False).reset_index()
        return tree_df

    def plot_train_graph(self):
        dataset_names = list(self.evals_result.keys())
        metrics = list(self.evals_result[dataset_names[0]].keys())
        n_rows = int(np.ceil(len(metrics)/2))
        plt.figure(figsize=(2*8, n_rows*4))
        for row, metric in enumerate(metrics):
            handles = []
            for column, dataset in enumerate(dataset_names):
                plt.subplot(n_rows, 2, row+1)
                plt.title(metric)
                [handle] = plt.plot(self.evals_result[dataset][metric])
                handles.append(handle)
            plt.legend(handles=handles, labels=dataset_names, loc='best')
        return plt

    def save_train_graph(self):
        self._save_dir.mkdir(parents=True, exist_ok=True)
        graph_plt = self.plot_train_graph()
        graph_plt.savefig(self._save_dir.joinpath("train_graph.png"))
        graph_plt.close()
    
    @property
    def save_dir(self):
        return self._save_dir

    @save_dir.setter
    def save_dir(self, value):
        self._save_dir = value

    def save(self):
        self.save_model()
        self.save_train_graph()
        self.save_test_graph()

    def save_test_graph(self, pos_label=1):
        graph_save_dir = self._save_dir.joinpath('graph')
        graph_save_dir.mkdir(parents=True, exist_ok=True)
        data_Xy = None
        title = 'oot score distribute'
        if self.oot_Xy is not None:
            data_Xy, title = self.oot_Xy, 'oot score distribute'
        elif self.test_Xy is not None:
            data_Xy, title = self.test_Xy, 'test score distribute'
        else:
            data_Xy, title = self.train_Xy, 'train score distribute'
        y_true = data_Xy.get_label()
        y_pred =self.predict_proba(data_Xy)[:, pos_label]

        _ = plt.hist(y_pred, bins=100)
        plt.title(title)
        plt.savefig(graph_save_dir.joinpath("score_distribute.png"))
        plt.close()

        cm = confusion_matrix(y_true, y_pred.round())
        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=pos_label, drop_intermediate=False)
        plot_roc_curve(fpr, tpr, thresholds)
        plt.savefig(graph_save_dir.joinpath("roc_curve.png"))
        plt.close()
        plot_ks_curve(fpr, tpr, thresholds)
        plt.savefig(graph_save_dir.joinpath("ks_curve.png"))
        plt.close()
        plot_fpr_tpr_curve(fpr, tpr, thresholds)
        plt.savefig(graph_save_dir.joinpath("fpr_tpr_curve.png"))
        plt.close()
        average_precision = average_precision_score(y_true, y_pred)
        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        plot_precision_recall_curve(precision, recall, average_precision)
        plt.savefig(graph_save_dir.joinpath("precision_recall_curve.png"))
        plt.close()
        plot_confusion_matrix(cm, classes=[0,1])
        plt.savefig(graph_save_dir.joinpath("confusion_matrix.png"))
        plt.close()
        score_bin_statistic_df = get_score_bin_statistic(y_true,y_pred)
        score_bin_statistic_df.to_csv(self._save_dir.joinpath("score_bin_statistic_df.csv"), index=False)
