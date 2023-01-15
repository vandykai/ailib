import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_consistent_length
from sklearn.metrics import make_scorer
from ailib.tools.utils_check import check_is_binary


def response_rate_by_threshold(y_true, uplift, treatment, group, uplift_bins):
    """Compute response rate (target mean in the control or treatment group) at each percentile.
    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        group (string, ['treatment', 'control']): Group type for computing response rate: treatment or control.
            * ``'treatment'``:
                Values equal 1 in the treatment column.
            * ``'control'``:
                Values equal 0 in the treatment column.
        bins (int): Determines the number of bins (and relative percentile) in the data. Default is 10.
        
    Returns:
        array (shape = [>2]), array (shape = [>2]), array (shape = [>2]):
        response rate at each percentile for control or treatment group,
        variance of the response rate at each percentile,
        group size at each percentile.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    group_types = ['treatment', 'control']

    if group not in group_types:
        raise ValueError(f'Response rate supports only group types in {group_types},'
                         f' got {group}.') 
    
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)
    trmnt_flag = 1 if group == 'treatment' else 0
    df = pd.DataFrame({'y_true':y_true, 'uplift':uplift, 'treatment':treatment, 'uplift_bins':uplift_bins})
    df = df[df['treatment']==trmnt_flag]
    df = df.groupby(['uplift_bins'], as_index=False, sort=True, dropna=True).agg(group_size=('y_true', 'count'), 
                                                                      response_rate=('y_true', 'mean'))
    response_rate = df['response_rate'].to_numpy()
    group_size = df['group_size'].to_numpy()
    df.loc[:, 'variance'] = np.multiply(response_rate, np.divide((1 - response_rate), group_size))

    return df


def weighted_score_average_uplift(y_true, uplift, treatment, bins=10):
    """Weighted average uplift.
    It is an average of uplift by percentile.
    Weights are sizes of the treatment group by percentile.
    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        bins (int): Determines the number of bins (and the relative percentile) in the data. Default is 10.
    Returns:
        float: Weighted average uplift.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)


    n_samples = len(y_true)

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(f'Bins should be positive integer.'
                         f' Invalid value bins: {bins}')

    if bins >= n_samples:
        raise ValueError(f'Number of bins = {bins} should be smaller than the length of y_true {n_samples}')

    uplift_bins = pd.cut(np.array(uplift), bins)

    df_trmnt = response_rate_by_threshold(
        y_true, uplift, treatment, group='treatment', uplift_bins=uplift_bins)

    df_ctrl = response_rate_by_threshold(
        y_true, uplift, treatment, group='control', uplift_bins=uplift_bins)

    uplift_scores = df_trmnt['response_rate'] - df_ctrl['response_rate']

    weighted_avg_uplift = np.dot(df_trmnt['group_size'].to_numpy(), uplift_scores) / np.sum(df_trmnt['group_size'].to_numpy())

    return weighted_avg_uplift


def uplift_by_threshold(y_true, uplift, treatment,
                         bins=10, std=False, total=False):
    """Compute metrics: uplift, group size, group response rate, standard deviation at each percentile.
    Metrics in columns and percentiles in rows of pandas DataFrame:
        - ``n_treatment``, ``n_control`` - group sizes.
        - ``response_rate_treatment``, ``response_rate_control`` - group response rates.
        - ``uplift`` - treatment response rate substract control response rate.
        - ``std_treatment``, ``std_control`` - (optional) response rates standard deviation.
        - ``std_uplift`` - (optional) uplift standard deviation.
    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        std (bool): If True, add columns with the uplift standard deviation and the response rate standard deviation.
            Default is False.
        total (bool): If True, add the last row with the total values. Default is False.
            The total uplift computes as a total response rate treatment - a total response rate control.
            The total response rate is a response rate on the full data amount.
        bins (int): Determines the number of bins (and the relative percentile) in the data. Default is 10.
    Returns:
        pandas.DataFrame: DataFrame where metrics are by columns and percentiles are by rows.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)

    n_samples = len(y_true)

    if not isinstance(total, bool):
        raise ValueError(f'Flag total should be bool: True or False.'
                         f' Invalid value total: {total}')

    if not isinstance(std, bool):
        raise ValueError(f'Flag std should be bool: True or False.'
                         f' Invalid value std: {std}')

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(f'Bins should be positive integer.'
                         f' Invalid value bins: {bins}')

    if bins >= n_samples:
        raise ValueError(f'Number of bins = {bins} should be smaller than the length of y_true {n_samples}')

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)
    uplift_bins = pd.cut(uplift, bins)
    df_trmnt = response_rate_by_threshold(
        y_true, uplift, treatment, group='treatment', uplift_bins=uplift_bins)

    df_ctrl = response_rate_by_threshold(
        y_true, uplift, treatment, group='control', uplift_bins=uplift_bins)
    result_df = pd.merge(df_trmnt, df_ctrl, on=['uplift_bins'], suffixes=('_trmnt', '_ctrl'))
    result_df.loc[:, 'uplift_scores'] = result_df['response_rate_trmnt'] - result_df['response_rate_ctrl']
    result_df.loc[:, 'uplift_variance'] =result_df['variance_trmnt'] + result_df['variance_ctrl']

    uplift_bins = result_df['uplift_bins'].to_numpy()
    n_trmnt = result_df['group_size_trmnt'].to_numpy()
    n_ctrl = result_df['group_size_ctrl'].to_numpy()
    response_rate_trmnt = result_df['response_rate_trmnt'].to_numpy()
    response_rate_ctrl = result_df['response_rate_ctrl'].to_numpy()
    uplift_scores = result_df['uplift_scores'].to_numpy()
    variance_trmnt = result_df['variance_trmnt'].to_numpy()
    variance_ctrl = result_df['variance_ctrl'].to_numpy()
    uplift_variance  = result_df['uplift_variance'].to_numpy()
    df = pd.DataFrame({
        'uplift_bins': uplift_bins,
        'n_treatment': n_trmnt,
        'n_control': n_ctrl,
        'response_rate_treatment': response_rate_trmnt,
        'response_rate_control': response_rate_ctrl,
        'uplift': uplift_scores
    })
    if total:
        df_trmnt_total = response_rate_by_threshold(
            y_true, uplift, treatment, group='treatment', uplift_bins=pd.cut(uplift, 1))

        df_ctrl_total = response_rate_by_threshold(
            y_true, uplift, treatment, group='control', uplift_bins=pd.cut(uplift, 1))
        result_total_df = pd.merge(df_trmnt_total, df_ctrl_total, on=['uplift_bins'], suffixes=('_trmnt', '_ctrl'))
        df.loc[-1, :] = ['total', result_total_df['group_size_trmnt'].to_numpy()[0], result_total_df['group_size_ctrl'].to_numpy()[0], result_total_df['response_rate_trmnt'].to_numpy()[0],
                         result_total_df['response_rate_ctrl'][0], result_total_df['response_rate_trmnt'][0] - result_total_df['response_rate_ctrl'][0]]

    if std:
        std_treatment = np.sqrt(variance_trmnt)
        std_control = np.sqrt(variance_ctrl)
        std_uplift = np.sqrt(uplift_variance)

        if total:
            std_treatment = np.append(std_treatment, np.sum(std_treatment))
            std_control = np.append(std_control, np.sum(std_control))
            std_uplift = np.append(std_uplift, np.sum(std_uplift))

        df.loc[:, 'std_treatment'] = std_treatment
        df.loc[:, 'std_control'] = std_control
        df.loc[:, 'std_uplift'] = std_uplift

    df = df \
        .set_index('uplift_bins', drop=True, inplace=False) \
        .astype({'n_treatment': 'int32', 'n_control': 'int32'})

    return df