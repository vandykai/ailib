import numpy as np
import pandas as pd
from sklearn.metrics import auc
from sklearn.utils.extmath import stable_cumsum
from sklearn.utils.validation import check_consistent_length
from sklearn.metrics import make_scorer
from ailib.tools.utils_check import check_is_binary


def make_uplift_scorer(metric_name, treatment, **kwargs):
    """Make uplift scorer which can be used with the same API as ``sklearn.metrics.make_scorer``.
    Args:
        metric_name (string): Name of desirable uplift metric. Raise ValueError if invalid.
        treatment (pandas.Series): A Series from original DataFrame which
            contains original index and treatment group column.
        kwargs (additional arguments): Additional parameters to be passed to metric func.
            For example: `negative_effect`, `strategy`, `k` or somtething else.
    Returns:
        scorer (callable): An uplift scorer with passed treatment variable (and kwargs, optionally) that returns a scalar score.
    Raises:
        ValueError: if `metric_name` does not present in metrics list.
        ValueError: if `treatment` is not a pandas Series.
    Example::
        from sklearn.model_selection import cross_validate
        from sklift.metrics import make_uplift_scorer
        # define X_cv, y_cv, trmnt_cv and estimator
        # Use make_uplift_scorer to initialize new `sklearn.metrics.make_scorer` object
        qini_scorer = make_uplift_scorer("qini_auc_score", trmnt_cv)
        # or pass additional parameters if necessary
        uplift50_scorer = make_uplift_scorer("uplift_at_k", trmnt_cv, strategy='overall', k=0.5)
        # Use this object in model selection functions
        cross_validate(estimator,
           X=X_cv,
           y=y_cv,
           fit_params={'treatment': trmnt_cv}
           scoring=qini_scorer,
        )
    """
    metrics_dict = {
        'uplift_auc_score': uplift_auc_score,
        'qini_auc_score': qini_auc_score,
        'uplift_at_k': uplift_at_k,
        'weighted_average_uplift': weighted_average_uplift,
    }

    if metric_name not in metrics_dict.keys():
        raise ValueError(
            f"'{metric_name}' is not a valid scoring value. "
            f"List of valid metrics: {list(metrics_dict.keys())}"
        )

    if not isinstance(treatment, pd.Series):
        raise TypeError("Expected pandas.Series in treatment vector, got %s" % type(treatment))

    def scorer(y_true, uplift, treatment_value, **kwargs):
        t = treatment_value.loc[y_true.index]
        return metrics_dict[metric_name](y_true, uplift, t, **kwargs)

    return make_scorer(scorer, treatment_value=treatment, **kwargs)


def uplift_curve(y_true, uplift, treatment):
    """Compute Uplift curve.
    For computing the area under the Uplift Curve, see :func:`.uplift_auc_score`.
    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.
    See also:
        :func:`.uplift_auc_score`: Compute normalized Area Under the Uplift curve from prediction scores.
        :func:`.perfect_uplift_curve`: Compute the perfect Uplift curve.
        :func:`.plot_uplift_curve`: Plot Uplift curves from predictions.
        :func:`.qini_curve`: Compute Qini curve.
    References:
        Devriendt, F., Guns, T., & Verbeke, W. (2020). Learning to rank for uplift modeling. ArXiv, abs/2002.05897.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)

    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    desc_score_indices = np.argsort(uplift, kind="mergesort")[::-1]
    y_true, uplift, treatment = y_true[desc_score_indices], uplift[desc_score_indices], treatment[desc_score_indices]

    y_true_ctrl, y_true_trmnt = y_true.copy(), y_true.copy()

    y_true_ctrl[treatment == 1] = 0
    y_true_trmnt[treatment == 0] = 0

    distinct_value_indices = np.where(np.diff(uplift))[0]
    threshold_indices = np.r_[distinct_value_indices, uplift.size - 1]

    num_trmnt = stable_cumsum(treatment)[threshold_indices]
    y_trmnt = stable_cumsum(y_true_trmnt)[threshold_indices]

    num_all = threshold_indices + 1

    num_ctrl = num_all - num_trmnt
    y_ctrl = stable_cumsum(y_true_ctrl)[threshold_indices]

    curve_values = (np.divide(y_trmnt, num_trmnt, out=np.zeros_like(y_trmnt), where=num_trmnt != 0) -
                    np.divide(y_ctrl, num_ctrl, out=np.zeros_like(y_ctrl), where=num_ctrl != 0)) * num_all

    if num_all.size == 0 or curve_values[0] != 0 or num_all[0] != 0:
        # Add an extra threshold position if necessary
        # to make sure that the curve starts at (0, 0)
        num_all = np.r_[0, num_all]
        curve_values = np.r_[0, curve_values]

    return num_all, curve_values


def perfect_uplift_curve(y_true, treatment):
    """Compute the perfect (optimum) Uplift curve.
    This is a function, given points on a curve.  For computing the
    area under the Uplift Curve, see :func:`.uplift_auc_score`.
    Args:
        y_true (1d array-like): Correct (true) binary target values.
        treatment (1d array-like): Treatment labels.
    Returns:
        array (shape = [>2]), array (shape = [>2]): Points on a curve.
    See also:
        :func:`.uplift_curve`: Compute the area under the Qini curve.
        :func:`.uplift_auc_score`: Compute normalized Area Under the Uplift curve from prediction scores.
        :func:`.plot_uplift_curve`: Plot Uplift curves from predictions.
    """

    check_consistent_length(y_true, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    y_true, treatment = np.array(y_true), np.array(treatment)

    cr_num = np.sum((y_true == 1) & (treatment == 0))  # Control Responders
    tn_num = np.sum((y_true == 0) & (treatment == 1))  # Treated Non-Responders

    # express an ideal uplift curve through y_true and treatment
    summand = y_true if cr_num > tn_num else treatment
    perfect_uplift = 2 * (y_true == treatment) + summand

    return uplift_curve(y_true, perfect_uplift, treatment)


def uplift_auc_score(y_true, uplift, treatment):
    """Compute normalized Area Under the Uplift Curve from prediction scores.
    By computing the area under the Uplift curve, the curve information is summarized in one number.
    For binary outcomes the ratio of the actual uplift gains curve above the diagonal to that of
    the optimum Uplift Curve.
    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
    Returns:
        float: Area Under the Uplift Curve.
    See also:
        :func:`.uplift_curve`: Compute Uplift curve.
        :func:`.perfect_uplift_curve`: Compute the perfect (optimum) Uplift curve.
        :func:`.plot_uplift_curve`: Plot Uplift curves from predictions.
        :func:`.qini_auc_score`: Compute normalized Area Under the Qini Curve from prediction scores.
    """

    check_consistent_length(y_true, uplift, treatment)
    check_is_binary(treatment)
    check_is_binary(y_true)
    y_true, uplift, treatment = np.array(y_true), np.array(uplift), np.array(treatment)

    x_actual, y_actual = uplift_curve(y_true, uplift, treatment)
    x_perfect, y_perfect = perfect_uplift_curve(y_true, treatment)
    x_baseline, y_baseline = np.array([0, x_perfect[-1]]), np.array([0, y_perfect[-1]])

    auc_score_baseline = auc(x_baseline, y_baseline)
    auc_score_perfect = auc(x_perfect, y_perfect) - auc_score_baseline
    auc_score_actual = auc(x_actual, y_actual) - auc_score_baseline

    return auc_score_actual / auc_score_perfect