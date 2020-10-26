import os

current_path = os.path.dirname(__file__)

def get_data(file_path):
    data = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            line_splitd = line.split("\t")
            if len(line_splitd) == 3:
                data.append(tuple(line_splitd))
            else:
                print(line)
    return data

# 数据格式 [(text_a, text_b, is_same_means), ...]
data_train = get_data(os.path.join(current_path, "data/train.txt"))
data_dev = get_data(os.path.join(current_path, "data/dev.txt"))
data_test = get_data(os.path.join(current_path, "data/test.txt"))