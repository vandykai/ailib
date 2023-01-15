import pandas as pd
import numpy as np

def check_df_label(data_df, key_columns, label_columns):
    """
    check whether data_df has different label_columns by key_columns
    """
    if not (type(label_columns) in [list, np.ndarray]):
        label_columns = [label_columns]
    def _check_label(grp):
        grp_labels = grp[label_columns].to_numpy().tolist()
        grp_labels = [tuple(it) for it in grp_labels]
        grp_labels = set(grp_labels)
        if len(grp_labels) > 1:
            return 1
        return 0
    issue_label = data_df.groupby(key_columns).apply(_check_label)
    issue_df = pd.merge(data_df, issue_label[issue_label==1].reset_index()[key_columns], on=key_columns, how='inner')
    return issue_df

def clean_df_label(data_df, key_columns, label_columns):
    """
    check whether data_df has different label_columns by key_columns
    """
    if not (type(label_columns) in [list, np.ndarray]):
        label_columns = [label_columns]
    def _check_label(grp):
        grp_labels = grp[label_columns].to_numpy().tolist()
        grp_labels = [tuple(it) for it in grp_labels]
        grp_labels = set(grp_labels)
        if len(grp_labels) > 1:
            return 1
        return 0
    issue_label = data_df.groupby(key_columns).apply(_check_label)
    normal_df = pd.merge(data_df, issue_label[issue_label==0].reset_index()[key_columns], on=key_columns, how='inner')
    return normal_df

def check_is_binary(array):
    """Checker if array consists of int or float binary values 0 (0.) and 1 (1.)
    Args:
        array (1d array-like): Array to check.
    """

    if not np.all(np.unique(array) == np.array([0, 1])):
        raise ValueError(f"Input array is not binary. "
                         f"Array should contain only int or float binary values 0 (or 0.) and 1 (or 1.). "
                         f"Got values {np.unique(array)}.")