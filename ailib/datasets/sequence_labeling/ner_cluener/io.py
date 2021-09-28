import os
import json

current_path = os.path.dirname(__file__)

def _load_data(file_path):
    examples = []
    with open(file_path, 'r') as f:
        idx = 0
        for line in f:
            line = json.loads(line.strip())
            text = line['text']
            label_entities = line.get('label', None)
            words = list(text)
            tags = ['O'] * len(words)
            if label_entities is not None:
                for key, value in label_entities.items():
                    for sub_name, sub_index in value.items():
                        for start_index, end_index in sub_index:
                            assert ''.join(words[start_index:end_index + 1]) == sub_name
                            if start_index == end_index:
                                tags[start_index] = 'S-' + key
                            else:
                                tags[start_index] = 'B-' + key
                                tags[start_index + 1:end_index + 1] = ['I-' + key] * (len(sub_name) - 1)
            examples.append((words, tags))
    return examples
def get_data():
    # 数据格式 [(words, tags), ...]
    data_train = _load_data(os.path.join(current_path, "data/train.json"))
    data_dev = _load_data(os.path.join(current_path, "data/dev.json"))
    data_test = _load_data(os.path.join(current_path, "data/test.json"))
    return data_train, data_dev, data_test