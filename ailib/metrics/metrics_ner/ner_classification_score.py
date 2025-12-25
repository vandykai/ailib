"""Score metric for NER Classification."""
from collections import Counter
from ailib.tools.utils_ner import get_entities
from ailib.tools.utils_dict import IdentityDict
import numpy as np
from ailib.metrics.base_metric import NerMetric

class SpanEntityScore(NerMetric):
    """Accuracy metric."""

    ALIAS = ['ner_span_score']

    def __init__(self, id2label=IdentityDict(), markup='bios'):
        self.id2label = id2label
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def __repr__(self) -> str:
        """:return: Formated string representation of the metric."""
        return f"{self.ALIAS[0]}"

    def __call__(self, y_true: list, y_pred: list) -> tuple:
        """
        Calculate score.

        Example:
            >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> metric = SpanEntityScore()
            >>> metric.update(y_true, y_pred)
            >>> metric.result()

        :param y_true: The ground true label of each sentence.
        :param y_pred: The predicted scores of each sentence.
        :return: Score Info.
        """
        origins, founds, rights = self._compute(y_true, y_pred) 

        return self._compute_score(origins, founds, rights)

    def _compute(self, y_true: list, y_pred: list) -> tuple:
        origins = []
        founds = []
        rights = []

        for y_true_label, y_pred_label in zip(y_true, y_pred):
            y_true_label = [self.id2label[item] for item in y_true_label]
            y_pred_label = [self.id2label[item] for item in y_pred_label]
            origins.extend(y_true_label)
            founds.extend(y_pred_label)
            for item_true, item_pred in zip(y_true_label, y_pred_label):
                if item_pred == item_true:
                    rights.append([item_pred])
        return origins, founds, rights
       
    def _compute_score(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self._compute_score(origin, found, right)
            class_info[type_] = {"precision": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4), "support":origin}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self._compute_score(origin, found, right)
        return {'_score':precision, 'acc': precision, 'recall': recall, 'f1': f1, "support":origin, 'class_info':class_info}

    def update(self, y_true: list, y_pred: list):
        '''
        y_true: [[],[],[],....]
        y_pred: [[],[],[],.....]

        :param y_true:
        :param y_pred:
        :return:
        Example:
            >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        origins, founds, rights = self._compute(y_true, y_pred)
        self.origins.extend(origins)
        self.founds.extend(founds)
        self.rights.extend(rights)

class SeqEntityScore(SpanEntityScore):

    ALIAS = ['ner_seq_score']

    def _compute(self, y_true: list, y_pred: list) -> tuple:
        origins = []
        founds = []
        rights = []

        for y_true_label, y_pred_label in zip(y_true, y_pred):
            y_true_entities = get_entities(y_true_label, self.id2label, self.markup)
            y_pred_entities = get_entities(y_pred_label, self.id2label, self.markup)
            origins.extend(y_true_entities)
            founds.extend(y_pred_entities)
            rights.extend([y_pred_entity for y_pred_entity in y_pred_entities if y_pred_entity in y_true_entities])
        return origins, founds, rights



