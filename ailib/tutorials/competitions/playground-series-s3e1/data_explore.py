import pandas as pd
from ailib import *
from autoviz.classify_method import data_cleaning_suggestions ,data_suggestions
from autoviz.AutoViz_Class import AutoViz_Class

train_df = pd.read_csv('/home/wandikai/work/testspace/playground-series-s3e1/train.csv')
test_df = pd.read_csv('/home/wandikai/work/testspace/playground-series-s3e1/test.csv')

data_cleaning_suggestions(train_df)

av = AutoViz_Class()
av.AutoViz('/home/wandikai/work/testspace/playground-series-s3e1/train.csv')