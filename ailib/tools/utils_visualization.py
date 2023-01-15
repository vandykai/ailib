import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from collections import OrderedDict
import torch.nn as nn
import torch
from matplotlib.pyplot import MultipleLocator
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix


def plot_confusion_matrix(cm, classes, save_path=None, title='Confusion Matrix'):
    plt.figure(figsize=(8, 4), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%d" % (c,), color='black', fontsize=15, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    if title:
        plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().yaxis.set_ticks_position('left')
    plt.gca().xaxis.set_label_position('top')
    plt.gca().yaxis.set_label_position('left')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    if save_path:
        plt.savefig(save_path, format='png')

def plot_heatmap(cm, classes, data_fmt='d'):
    df_cm = pd.DataFrame(cm, index = classes, columns =classes)
    return sns.heatmap(df_cm, annot=True, fmt=data_fmt)

def plot_roc_curve(fpr: list, tpr: list, thresholds: list):
    ks = np.max(tpr-fpr)
    ks_pos = np.argmax(tpr-fpr)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot([0, 1-ks], [ks, 1], 'r--')
    plt.text(0, ks, round(ks, 4), ha='right', va='center', fontsize=10)
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve (area = {round(roc_auc, 4)}, ks={round(ks, 4)}, thresholds={thresholds[ks_pos]})')
    #plt.legend(loc="lower right")

def plot_ks_curve(fpr: list, tpr: list, thresholds: list):
    ks = np.max(tpr-fpr)
    ks_pos = np.argmax(tpr-fpr)
    plt.plot(thresholds, tpr-fpr)
    #plt.plot([0, 1], [ks, ks], 'k--')
    plt.text(0, ks, round(ks, 4), ha='right', va='center', fontsize=10)
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.05))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, ks])
    plt.xlabel('thresholds')
    plt.ylabel('KS value')
    plt.title(f'KS curve (ks={round(ks, 4)}, thresholds={thresholds[ks_pos]})')
    #plt.legend(loc="lower right")

def plot_fpr_tpr_curve(fpr: list, tpr: list, thresholds: list):
    [fpr_plot] = plt.plot(thresholds, fpr, 'b')
    [tpr_plot] = plt.plot(thresholds, tpr, 'r')
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(0.1))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('thresholds')
    plt.ylabel('tpr_fpr value')
    plt.title(f'tpr fpr curve')
    plt.legend(handles=[fpr_plot, tpr_plot], labels=['fpr','tpr'], loc='best')

def plot_precision_recall_curve(precision, recall, average_precision):
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall: {0:0.6f}'.format(average_precision))

def plot_cls_result(y_true: list, y_pred: list, pos_label=1):
    y_true = np.array(y_true, dtype=np.int8)
    y_pred = np.array(y_pred, dtype=np.float64)
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, pos_label]
    cm = confusion_matrix(y_true, y_pred.round())
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=pos_label, drop_intermediate=False)
    plot_roc_curve(fpr, tpr, thresholds)
    plt.show()
    plot_ks_curve(fpr, tpr, thresholds)
    plt.show()
    plot_fpr_tpr_curve(fpr, tpr, thresholds)
    plt.show()
    average_precision = average_precision_score(y_true, y_pred)
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    plot_precision_recall_curve(precision, recall, average_precision)
    plt.show()
    plot_confusion_matrix(cm, classes=[0,1])
    plt.show()

def plot_cls_auc(y_true: list, y_pred: list, pos_label=1):
    y_true = np.array(y_true, dtype=np.int8)
    y_pred = np.array(y_pred, dtype=np.float64)
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, pos_label]
    cm = confusion_matrix(y_true, y_pred.round())
    fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=pos_label, drop_intermediate=False)
    plot_roc_curve(fpr, tpr, thresholds)

def print_decisition_path(text_feature, clf, text_feature_name):
    node_indicator = clf.decision_path(text_feature)
    leave_id = clf.apply(text_feature)
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_index = node_indicator.indices[node_indicator.indptr[0]:
                                    node_indicator.indptr[0 + 1]]
    for node_id in node_index:
        if leave_id[0] == node_id:
            continue

        if (text_feature[0][feature[node_id]] <= threshold[node_id]):
            threshold_sign = "<="
        else:
            threshold_sign = ">"

        print("decision id node %s : (text_feature[%s, %s] (= %s) %s %s)"
              % (node_id,
                 0,
                 text_feature_name[feature[node_id]],
                 text_feature[0][feature[node_id]],
                 threshold_sign,
                 threshold[node_id]))

def plot_dict_bar(dict_value, y_type='percent', figsize='auto', reverse=True, **kwargs):
    if figsize=='auto':
        figsize = (4, int(len(dict_value)/5))
        fig = plt.figure(figsize=figsize)
    elif figsize:
        fig = plt.figure(figsize=figsize)
    dict_value = sorted(dict_value.items(), key=lambda x:float(str(x[0]).split('-')[0]))
    x = [str(it[0]) for it in dict_value]
    y = [it[1] for it in dict_value]
    plt.title(f"total num:{sum(y)}")
    assert y_type in ['cumsum', 'count', 'percent']
    if y_type == 'cumsum':
        y = np.cumsum(y)/sum(y)
    elif y_type == 'percent':
        y = np.array(y)/sum(y)

    num=np.arange(len(y))
    if reverse:
        plt.ylim(min(num)-1,max(num)+1)
        h = plt.barh(x, y, **kwargs)
        plt.bar_label(h)
    else:
        plt.xlim(min(num)-1,max(num)+1)
        h = plt.bar(x, y, **kwargs)
        plt.bar_label(h)
    return h

def plot_dict_line(dict_value, y_type='cumsum', figsize=(4,4), reverse=True, **kwargs):
    if figsize:
        fig = plt.figure(figsize=figsize)
    dict_value = sorted(dict_value.items(), key=lambda x:float(str(x[0]).split('-')[0]))
    x = [str(it[0]) for it in dict_value]
    y = [it[1] for it in dict_value]
    if y_type == 'cumsum':
        y = np.cumsum(y)/sum(y)
    elif y_type == 'percent':
        y = np.array(y)/sum(y)
    num=np.arange(len(y))
    
    if reverse:
        plt.ylim(min(num)-1,max(num)+1)
        h, = plt.plot(y, x, **kwargs)
    else:
        plt.xlim(min(num)-1,max(num)+1)
        h, = plt.plot(x, y, **kwargs)
    return h


def get_score_bin_statistic(y_true: list, y_pred: list, pos_label=1, bins=10, actual_pos_rate=None):
    y_true = np.array(y_true, dtype=np.int8)
    y_pred = np.array(y_pred, dtype=np.float64)
    if len(y_pred.shape) == 2:
        y_pred = y_pred[:, pos_label]
    score_bin = pd.cut(y_pred, bins)
    result_df = pd.DataFrame({'score_bin':score_bin, 'y_true':y_true, 'y_pred':y_pred})
    result_df = result_df.groupby(['score_bin'], as_index=False, sort=False, dropna=True).agg(sample_num=('y_true', 'count'), 
                                                                      pos_sample_num=('y_true', lambda x:np.sum(x==pos_label)))
    
    if actual_pos_rate is not None:
        global_pos_rate = result_df['pos_sample_num'].sum()/result_df['sample_num'].sum()
        neg_global_pos_rate_rate = (1/actual_pos_rate-1)/(1/global_pos_rate-1)
        result_df['sample_num'] = (result_df['sample_num'] - result_df['pos_sample_num'])*neg_global_pos_rate_rate + result_df['pos_sample_num']
    result_df.sort_values(by=['score_bin'], ascending=False, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    result_df['sample_cumsum'] = result_df['sample_num'].cumsum()
    result_df['pos_sample_cumsum'] = result_df['pos_sample_num'].cumsum()
    result_df['pos_sample_rate'] = result_df['pos_sample_num']/result_df['sample_num']
    result_df['precision'] = result_df['pos_sample_cumsum']/result_df['sample_cumsum']
    result_df['recall/tpr'] = result_df['pos_sample_cumsum']/result_df['pos_sample_num'].sum()
    result_df['fpr'] = (result_df['sample_cumsum']-result_df['pos_sample_cumsum'])/(result_df['sample_num'].sum()-result_df['pos_sample_num'].sum())
    result_df['ks'] = result_df['recall/tpr'] - result_df['fpr']
    result_df['lift'] = (result_df['pos_sample_num']/result_df['sample_num'])/(result_df['pos_sample_num'].sum()/result_df['sample_num'].sum())
    result_df['lift_cumsum'] = result_df['precision']/(result_df['pos_sample_num'].sum()/result_df['sample_num'].sum())
    return result_df

def plot_time_distribute(df, date_key, label_key, pos_label=1, figsize='auto'):
    '''
    打印：全量/正样本随时间的分布，
    :param df:
    :param date_key:
    :param label_key:
    :param figsize:
    :return:
    Example:
        >>> plot_time_distribute(df, 'loan_date', 'label')
    '''
    dict_value = df[date_key].value_counts().to_dict()
    if figsize=='auto':
        figsize = (15, int(len(dict_value)/5))
        fig = plt.figure(figsize=figsize)
    elif figsize:
        fig = plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plot_dict_bar(dict_value, figsize=None, y_type='cumsum')
    plt.subplot(1, 2, 2)
    pos_df = df[df[label_key]==pos_label]
    plot_dict_bar(pos_df[date_key].value_counts().to_dict(), figsize=None, y_type='cumsum')
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)
    return plt

def model_summary(model, *inputs, batch_size=-1, show_input=True):
    '''
    打印模型结构信息
    :param model:
    :param inputs:
    :param batch_size:
    :param show_input:
    :return:
    Example:
        >>> print("model summary info: ")
        >>> for step,batch in enumerate(train_data):
        >>>     model_summary(self.model,*batch,show_input=True)
        >>>     break
    '''

    def register_hook(module):
        def hook(module, input, output=None):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = f"{class_name}-{module_idx + 1}"
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size

            if show_input is False and output is not None:
                if isinstance(output, (list, tuple)):
                    for out in output:
                        if isinstance(out, torch.Tensor):
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out.size())[1:]
                            ][0]
                        else:
                            summary[m_key]["output_shape"] = [
                                [-1] + list(out[0].size())[1:]
                            ][0]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model)):
            if show_input is True:
                hooks.append(module.register_forward_pre_hook(hook))
            else:
                hooks.append(module.register_forward_hook(hook))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)
    model(*inputs)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("-----------------------------------------------------------------------")
    if show_input is True:
        line_new = f"{'Layer (type)':>25}  {'Input Shape':>25} {'Param #':>15}"
    else:
        line_new = f"{'Layer (type)':>25}  {'Output Shape':>25} {'Param #':>15}"
    print(line_new)
    print("=======================================================================")

    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        if show_input is True:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["input_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )
        else:
            line_new = "{:>25}  {:>25} {:>15}".format(
                layer,
                str(summary[layer]["output_shape"]),
                "{0:,}".format(summary[layer]["nb_params"]),
            )

        total_params += summary[layer]["nb_params"]
        if show_input is True:
            total_output += np.prod(summary[layer]["input_shape"])
        else:
            total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]

        print(line_new)

    print("=======================================================================")
    print(f"Total params: {total_params:0,}")
    print(f"Trainable params: {trainable_params:0,}")
    print(f"Non-trainable params: {(total_params - trainable_params):0,}")
    print("-----------------------------------------------------------------------")