import numpy as np
from collections import Counter
from ailib.tools.utils_dict import IdentityDict
from ailib.metrics.base_metric import ClassificationMultiLabelMetric
from ailib.metrics.utils import sigmoid

class ClassificationMultiLabelScore(ClassificationMultiLabelMetric):

    ALIAS = ['score']

    def __init__(self, id2label=IdentityDict(), threshold = 0.5):
        super().__init__()
        self.id2label = id2label
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.origins = []
        self.mix_origins = []
        self.founds = []
        self.rights = []

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}@{self.threshold}"

    def __call__(self, y_true: list, y_pred: list) -> float:
        """
        Calculate score.

        Example:
            >>> y_true = [[0, 0, 1, 0], [0, 1, 0, 1]]
            >>> y_pred = [[0, 0.4, 1.7, 0], [0, 0.5, 0.1, 0.6]]
            >>> metric = ClassificationMultiLabelScore()
            >>> metric.update(y_true, y_pred)
            >>> metric.result()

        :param y_true: The ground true label of each document.
        :param y_pred: The predicted scores of each document.
        :return: Score Info.
        """
        origins, founds, rights, mix_origins = self._compute(y_true, y_pred) 

        return self._compute_score(origins, founds, rights, mix_origins)

    def _compute_subject_score(self, origin, found, right):
        precision = 0 if found == 0 else (right / found)
        recall = 0 if origin == 0 else (right / origin)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

    def _compute_score(self, origins, founds, rights, mix_origins):
        origin = list(map(len, origins))
        right = list(map(len, rights))
        found = list(map(len, founds))
        mix_origin = list(map(len, mix_origins))
        jaccard = np.divide(right, mix_origin, out=np.zeros_like(right, dtype=np.float64), where=np.array(mix_origin)!=0)
        precision = np.divide(right, found, out=np.zeros_like(right, dtype=np.float64), where=np.array(found)!=0)   
        recall = np.divide(right, origin, out=np.zeros_like(right, dtype=np.float64), where=np.array(origin)!=0) 
        return jaccard.mean().item(), precision.mean().item(), recall.mean().item()

    def _compute(self, y_true: list, y_pred: list) -> list:
        origins = []
        founds = []
        rights = []
        mix_origins = []
        y_true = np.array(y_true, dtype=np.int8)
        y_pred = np.array(y_pred, dtype=np.float64)
        y_pred = sigmoid(y_pred)
        y_pred = (y_pred > self.threshold).astype(np.int8)
        for label, pred_label in zip(y_true, y_pred):
            label = sum(np.argwhere(label==1).tolist(), [])
            pred_label = sum(np.argwhere(pred_label==1).tolist(), [])
            label = [self.id2label[item] for item in label]
            pred_label = [self.id2label[item] for item in pred_label]
            origins.append(label)
            founds.append(pred_label)
            rights.append(list(set(label)&set(pred_label)))
            mix_origins.append(list(set(label)|set(pred_label)))
        return origins, founds, rights, mix_origins

    def update(self, y_true: list, y_pred: list):
        origins, founds, rights, mix_origins = self._compute(y_true, y_pred)
        self.origins.extend(origins)
        self.founds.extend(founds)
        self.rights.extend(rights)
        self.mix_origins.extend(mix_origins)

    def result(self):
        class_info = {}
        origin_counter = Counter([ex for item in self.origins for ex in item])
        found_counter = Counter([ex for item in self.founds for ex in item])
        right_counter = Counter([ex for item in self.rights for ex in item])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            precision, recall, f1 = self._compute_subject_score(origin, found, right)
            class_info[type_] = {"precision": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4), "support":origin}

        jaccard, precision, recall = self._compute_score(self.origins, self.founds, self.rights, self.mix_origins)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return {'_score':jaccard, 'jaccard': round(jaccard, 4), "precision":round(precision, 4), "recall":round(recall, 4), "f1":round(f1, 4), 'class_info':class_info}



