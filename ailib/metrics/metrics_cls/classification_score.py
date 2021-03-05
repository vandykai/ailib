import numpy as np
from collections import Counter
from ailib.tools.utils_dict import IdentityDict

class ClassificationScore(object):

    def __init__(self, id2label=IdentityDict()):
        self.id2label = id2label
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter(self.origins)
        found_counter = Counter(self.founds)
        right_counter = Counter(self.rights)
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"precision": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4), "support":origin}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1, "support":origin}, class_info

    def update(self, y_true: list, y_pred: list):
        '''

        :param y_true: [1, 3]
        :param y_pred: [[0, 1, 0, 0], [0, 0.3, 0.1, 0.6]]
        :return:
        Example:
            >>> import numpy as np
            >>> y_true = [1, 3]
            >>> y_pred = [[0, 1, 0, 0], [0, 0.3, 0.1, 0.6]]
        '''
        y_true = [self.id2label[item] for item in y_true]
        y_pred = np.argmax(y_pred, axis=1)
        y_pred = [self.id2label[item] for item in y_pred]

        self.origins.extend(y_true)
        self.founds.extend(y_pred)
        for label, pred_label in zip(y_true, y_pred):
            if label == pred_label:
                self.rights.append(label)




