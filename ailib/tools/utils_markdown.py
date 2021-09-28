import typing
import prettytable
import numpy as np
import pandas as pd
from collections import Counter

def df2markdown(data_df: pd.DataFrame):
    # 将DataFrame转换为markdown表格格式
    table = prettytable.PrettyTable(data_df.columns.tolist())
    table.set_style(prettytable.MARKDOWN)
    table.add_rows(data_df.values)
    print(table)
    return table

def list2markdown(data_list: list, columns: typing.Union[int, list]):
    # 将网页复制的表格转换为markdown表格格式
    # 若columns为int类型, 则默认头在data_list首行
    if type(columns) == int:
        columns , data_list= data_list[:columns], data_list[columns:]
    table = prettytable.PrettyTable(columns)
    table.set_style(prettytable.MARKDOWN)
    table.add_rows(np.array(data_list).reshape(-1, len(columns)))
    print(table)
    return table

def label2markdown(label_list: list, columns: list = ['标签','数量','百分比']):
    # 统计标签信息, 转换为markdown表格格式
    counter = Counter()
    counter.update(label_list)
    counter = sorted(counter.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

    table = prettytable.PrettyTable(['标签','数量','百分比'])
    table.set_style(prettytable.MARKDOWN)
    sum_value = 0
    for key, value in counter:
        sum_value += value
    for key, value in counter:
        table.add_row([key, value, "{0:.2f}%".format(value*100/sum_value)])
    print(table)
