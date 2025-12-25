import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_consistent_length
from sklearn.metrics import make_scorer
from ailib.tools.utils_check import check_is_binary


def treatment_balance_curve(uplift, treatment, winsize):
    """Compute the treatment balance curve: proportion of treatment group in the ordered predictions.
    Args:
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        winsize(int): Size of the sliding window for calculating the balance between treatment and control.
    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.
    """

    check_consistent_length(uplift, treatment)
    check_is_binary(treatment)
    uplift, treatment = np.array(uplift), np.array(treatment)

    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]

    treatment = treatment[desc_score_indices]

    balance = np.convolve(treatment, np.ones(winsize), 'valid') / winsize
    idx = np.linspace(1, 100, len(balance))
    return idx, balance


def average_squared_deviation(y_true_train, uplift_train, treatment_train, y_true_val,
                              uplift_val, treatment_val, strategy='overall', bins=10):
    """Compute the average squared deviation.
    The average squared deviation (ASD) is a model stability metric that shows how much the model overfits
    the training data. Larger values of ASD mean greater overfit.
    Args:
        y_true_train (1d array-like): Correct (true) target values for training set.
        uplift_train (1d array-like): Predicted uplift for training set, as returned by a model.
        treatment_train (1d array-like): Treatment labels for training set.
        y_true_val (1d array-like): Correct (true) target values for validation set.
        uplift_val (1d array-like): Predicted uplift for validation set, as returned by a model.
        treatment_val (1d array-like): Treatment labels for validation set.
        strategy (string, ['overall', 'by_group']): Determines the calculating strategy. Default is 'overall'.
            * ``'overall'``:
                The first step is taking the first k observations of all test data ordered by uplift prediction
                (overall both groups - control and treatment) and conversions in treatment and control groups
                calculated only on them. Then the difference between these conversions is calculated.
            * ``'by_group'``:
                Separately calculates conversions in top k observations in each group (control and treatment)
                sorted by uplift predictions. Then the difference between these conversions is calculated
        bins (int): Determines the number of bins (and the relative percentile) in the data. Default is 10.
    Returns:
        float: average squared deviation
    References:
        René Michel, Igor Schnakenburg, Tobias von Martens. Targeting Uplift. An Introduction to Net Scores.
    """
    check_consistent_length(y_true_train, uplift_train, treatment_train)
    check_is_binary(treatment_train)

    check_consistent_length(y_true_val, uplift_val, treatment_val)
    check_is_binary(treatment_val)

    strategy_methods = ['overall', 'by_group']

    n_samples_train = len(y_true_train)
    n_samples_val = len(y_true_val)
    min_n_samples = min(n_samples_train, n_samples_val)

    if strategy not in strategy_methods:
        raise ValueError(
            f'Response rate supports only calculating methods in {strategy_methods},'
            f' got {strategy}.')

    if not isinstance(bins, int) or bins <= 0:
        raise ValueError(
            f'Bins should be positive integer. Invalid value bins: {bins}')

    if bins >= min_n_samples:
        raise ValueError(
            f'Number of bins = {bins} should be smaller than the length of y_true_train {n_samples_train}'
            f'and length of y_true_val {n_samples_val}')

    uplift_by_percentile_train = uplift_by_percentile(y_true_train, uplift_train, treatment_train,
                                                      strategy=strategy, bins=bins)
    uplift_by_percentile_val = uplift_by_percentile(y_true_val, uplift_val, treatment_val,
                                                    strategy=strategy, bins=bins)
    
    return np.mean(np.square(uplift_by_percentile_train['uplift'] - uplift_by_percentile_val['uplift']))


def max_prof_uplift(df_sorted, treatment_name, churn_name, pos_outcome, benefit, c_incentive, c_contact, a_cost=0):
  """Compute the maximum profit generated from an uplift model decided campaign
    This can be visualised by plotting plt.plot(perc, cumulative_profit)
    Args:
      df_sorted (pandas dataframe): dataframe with descending uplift predictions for each customer (i.e. highest 1st)
      treatment_name (string): column name of treatment columm (assuming 1 = treated)   
      churn_name (string): column name of churn column
      pos_outcome (int or float): 1 or 0 value in churn column indicating a positive outcome (i.e. purchase = 1, whereas churn = 0)
      benefit (int or float): the benefit of retaining a customer (e.g., the average customer lifetime value)
      c_incentive (int or float): the cost of the incentive if a customer accepts the offer
      c_contact (int or float): the cost of contacting a customer regardless of conversion
      a_cost (int or float): the fixed administration cost for the campaign
    Returns:
      1d array-like: the incremental increase in x, for plotting
      1d array-like: the cumulative profit per customer 
    References:
        Floris Devriendt, Jeroen Berrevoets, Wouter Verbeke. Why you should stop predicting customer churn and start using uplift models.
  """
#   VARIABLES

#       n_ct0             no. people not treated
#       n_ct1             no. people treated

#       n_y1_ct0          no. people not treated with +ve outcome
#       n_y1_ct1          no. people treated with +ve outcome

#       r_y1_ct0          mean of not treated people with +ve outcome
#       r_y1_ct1          mean of treated people with +ve outcome

#       cs                cumsum() of each variable

  n_ct0 = np.where(df_sorted[treatment_name] == 0, 1, 0)
  cs_n_ct0 = pd.Series(n_ct0.cumsum())

  n_ct1 = np.where(df_sorted[treatment_name] == 1, 1, 0)
  cs_n_ct1 = pd.Series(n_ct1.cumsum())

  if pos_outcome == 0:
    n_y1_ct0 = np.where((df_sorted[treatment_name] == 0) & (df_sorted[churn_name] == 0), 1, 0)
    n_y1_ct1 = np.where((df_sorted[treatment_name] == 1) & (df_sorted[churn_name] == 0), 1, 0)  

  elif pos_outcome == 1:
    n_y1_ct0 = np.where((df_sorted[treatment_name] == 0) & (df_sorted[churn_name] == 1), 1, 0)
    n_y1_ct1 = np.where((df_sorted[treatment_name] == 1) & (df_sorted[churn_name] == 1), 1, 0)
    
  cs_n_y1_ct0 = pd.Series(n_y1_ct0.cumsum())
  cs_n_y1_ct1 = pd.Series(n_y1_ct1.cumsum())

  cs_r_y1_ct0 = (cs_n_y1_ct0 / cs_n_ct0).fillna(0)
  cs_r_y1_ct1 = (cs_n_y1_ct1 / cs_n_ct1).fillna(0)

  cs_uplift = cs_r_y1_ct1 - cs_r_y1_ct0

  # Dataframe of all calculated variables
  df = pd.concat([cs_n_ct0,cs_n_ct1,cs_n_y1_ct0,cs_n_y1_ct1, cs_r_y1_ct0, cs_r_y1_ct1, cs_uplift], axis=1)
  df.columns = ['cs_n_ct0', 'cs_n_ct1', 'cs_n_y1_ct0', 'cs_n_y1_ct1', 'cs_r_y1_c0', 'cs_r_y1_ct1', 'cs_uplift']

  x = cs_n_ct0 + cs_n_ct1
  max = cs_n_ct0.max() + cs_n_ct1.max()

  t_profit = (x * cs_uplift * benefit) - (c_incentive * x * cs_r_y1_ct1) - (c_contact * x) - a_cost
  perc = x / max
  cumulative_profit = t_profit / max

  return perc, cumulative_profit