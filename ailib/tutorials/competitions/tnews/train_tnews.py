from ailib.datasets.classification.tnews_short_text_classification.io import get_data

data_train, data_dev, data_test = get_data()
print(data_train[1])