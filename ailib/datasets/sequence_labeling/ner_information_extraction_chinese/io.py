import os

current_path = os.path.dirname(__file__)

def _load_data(file_path):
    examples = []
    with open(file_path) as f:
        words = []
        tags = []
        for line in f:
            line = line.strip()
            line_splitd = line.split()
            if len(line_splitd) == 2:
                words.append(line_splitd[0])
                tags.append(line_splitd[1])
            else:
                examples.append((words, tags))
                words = []
                tags = []
                if len(line_splitd) != 0:
                    print(line)
    return examples
def get_data():
    # 数据格式 [(words, tags), ...]
    data_train = _load_data(os.path.join(current_path, "data/example.train"))
    data_dev = _load_data(os.path.join(current_path, "data/example.dev"))
    data_test = _load_data(os.path.join(current_path, "data/example.test"))
    return data_train, data_dev, data_test