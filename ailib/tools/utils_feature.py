import numpy as np
from collections import Counter
import math
from scipy.sparse.csr import csr_matrix
from tqdm.auto import tqdm

def IV(feature : np.array, target : np.array, pos_label : int):
    pos_value = feature[np.where(target==pos_label)[0]]
    neg_value = feature[np.where(target!=pos_label)[0]]
    len_pos = len(pos_value)
    len_neg = len(neg_value)
    pos_counter = Counter(pos_value)
    neg_counter = Counter(neg_value)
    value_list = set(pos_counter.keys()) | set(neg_counter.keys())
    iv = 0
    for value in value_list:
        pos_rate = pos_counter.get(value, 1) / len_pos
        neg_rate = neg_counter.get(value, 1) / len_neg
        iv += (pos_rate - neg_rate) * math.log(pos_rate / neg_rate, 2)
    return iv

# def get_sparse_feature_IV(features : csr_matrix, target : np.array, pos_label : int):
#     iv_list = []
#     pos_index = np.where(target==pos_label)[0]
#     neg_index = np.where(target!=pos_label)[0]
#     len_pos = len(pos_index)
#     len_neg = len(neg_index)
#     pos_values = features[pos_index, :]
#     neg_values = features[neg_index, :]
#     for i in tqdm(range(features.shape[1])):
#         pos_value = pos_values[:, i].toarray().reshape(-1)
#         neg_value = neg_values[:, i].toarray().reshape(-1)
#         pos_counter = Counter(pos_value)
#         neg_counter = Counter(neg_value)
#         value_list = set(pos_counter.keys()) | set(neg_counter.keys())
#         iv = 0
#         for value in value_list:
#             pos_rate = pos_counter.get(value, 1) / len_pos
#             neg_rate = neg_counter.get(value, 1) / len_neg
#             iv += (pos_rate - neg_rate) * math.log(pos_rate / neg_rate, 2)
#         iv_list.append(iv)
#     return iv_list

def get_sparse_feature_IV(features : csr_matrix, target : np.array, pos_label : int, batch_size : int = 512):
    iv_list = []
    pos_index = np.where(target==pos_label)[0]
    neg_index = np.where(target!=pos_label)[0]
    len_pos = len(pos_index)
    len_neg = len(neg_index)
    pos_values = features[pos_index, :]
    neg_values = features[neg_index, :]
    feature_num = features.shape[1]

    with tqdm(total=feature_num) as pbar:
        for batch_idx in range(math.ceil(feature_num/batch_size)):
            pos_values_batch = pos_values[:, batch_idx*batch_size:(batch_idx+1)*batch_size]
            neg_values_batch = neg_values[:, batch_idx*batch_size:(batch_idx+1)*batch_size]
            for i in range(pos_values_batch.shape[1]):
                pos_value = pos_values_batch[:, i].toarray().reshape(-1)
                neg_value = neg_values_batch[:, i].toarray().reshape(-1)
                pos_counter = Counter(pos_value)
                neg_counter = Counter(neg_value)
                value_list = set(pos_counter.keys()) | set(neg_counter.keys())
                iv = 0
                for value in value_list:
                    pos_rate = pos_counter.get(value, 1) / len_pos
                    neg_rate = neg_counter.get(value, 1) / len_neg
                    iv += (pos_rate - neg_rate) * math.log(pos_rate / neg_rate, 2)
                iv_list.append(iv)
                pbar.update(1)
    return iv_list