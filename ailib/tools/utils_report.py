
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_curve)
from ailib.tools.utils_visualization import (get_score_bin_statistic,
                                             plot_confusion_matrix,
                                             plot_fpr_tpr_curve, plot_ks_curve,
                                             plot_precision_recall_curve,
                                             plot_roc_curve,
                                             precision_recall_curve)

def save_classification_report(save_path, y_true, y_pred, pos_label=1, y_pred_min=None, y_pred_max=None):
    save_path.mkdir(parents=True, exist_ok=True)
    y_pred_min = y_pred_min if y_pred_min is not None else y_pred.min()
    y_pred_max = y_pred_max if y_pred_max is not None else y_pred.max()

    title = f'score distribute'
    _ = plt.hist(y_pred, bins=100)
    plt.title(title)
    plt.savefig(save_path.joinpath("score_distribute.png"))
    plt.close()

    cm = confusion_matrix(y_true, y_pred.round())
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=pos_label, drop_intermediate=False)
    plot_roc_curve(fpr, tpr, thresholds)
    plt.savefig(save_path.joinpath("roc_curve.png"))
    plt.close()
    plot_ks_curve(fpr, tpr, thresholds)
    plt.savefig(save_path.joinpath("ks_curve.png"))
    plt.close()
    plot_fpr_tpr_curve(fpr, tpr, thresholds)
    plt.savefig(save_path.joinpath("fpr_tpr_curve.png"))
    plt.close()
    average_precision = average_precision_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plot_precision_recall_curve(precision, recall, average_precision)
    plt.savefig(save_path.joinpath("precision_recall_curve.png"))
    plt.close()
    plot_confusion_matrix(cm, classes=[0,1])
    plt.savefig(save_path.joinpath("confusion_matrix.png"))
    plt.close()
    frequence_score_bin_statistic_df = get_score_bin_statistic(y_true,y_pred, bins_type='frequence')
    frequence_score_bin_statistic_df.to_csv(save_path.joinpath(f"frequence_score_bin_statistic_df.csv"), index=False)

    distince_score_bin_statistic_df = get_score_bin_statistic(y_true,y_pred, bins_type='distance', bins=np.arange(y_pred_min,y_pred_max+1e-5, 0.1, ))
    distince_score_bin_statistic_df.to_csv(save_path.joinpath(f"distince_score_bin_statistic_df.csv"), index=False)