import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def plot_confusion_matrix(cm, classes, save_path=None, title='Confusion Matrix'):
    plt.figure(figsize=(16, 8), dpi=100)
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
    plt.show()

def plot_heatmap(cm, classes, data_fmt='d'):
    df_cm = pd.DataFrame(cm, index = classes, columns =classes)
    return sns.heatmap(df_cm, annot=True, fmt=data_fmt)

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