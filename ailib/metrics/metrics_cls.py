import torch
from collections import Counter

class ClassificationScore(object):
    def __init__(self, id2label={}):
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
            class_info[type_] = {"acc": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'acc': precision, 'recall': recall, 'f1': f1}, class_info

    def update(self, labels, pred_labels):
        '''
        labels: [c1,c2,c3,....]
        pred_labels: [c1,c2,c3,....]

        :param labels:
        :param pred_labels:
        :return:
        Example:
            >>> labels = ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7']
            >>> pred_labels = ['c1', 'c3', 'c2', 'c4', 'c5', 'c6', 'c7']
        '''
        labels = [self.id2label[item] for item in labels]
        pre_labels = [self.id2label[item] for item in pred_labels]
        self.origins.extend(labels)
        self.founds.extend(pre_labels)
        for label, pre_label in zip(labels, pre_labels):
            if label == pre_label:
                self.rights.append(label)




