import enum
import numpy as np

class IdentityDict(dict):
    def __missing__(self, key):
        return key

def dict_DFS(data, func, extra_tag=None):
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = dict_DFS(v, func, k)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            data[index] = dict_DFS(item, func)
    data = func(extra_tag, data)
    return data

def dict_BFS(data, func, extra_tag=None):
    data = func(extra_tag, data)
    if isinstance(data, dict):
        for k, v in data.items():
            data[k] = dict_BFS(v, func, k)
    elif isinstance(data, list):
        for index, item in enumerate(data):
            data[index] = dict_BFS(data[index], func)
    return data

def dict_get(data, key):
    value = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k == key:
                value.append(v)
            else:
                value.extend(dict_get(v, key))
    elif isinstance(data, list):
        for item in data:
            value.extend(dict_get(item, key))
    return value

def get_df_dict(data_df, key_columns, value_columns, different='drop'):
    assert different in ('last', 'first', 'all', 'drop')
    def _get_df_dict(grp):
        grp_labels = grp[value_columns].to_numpy().tolist()
        grp_labels = sorted(grp_labels)
        if (type(value_columns) in [list, np.ndarray]):
            grp_labels = [tuple(it) for it in grp_labels]
        if different=='all':
            grp_labels = [list(set(grp_labels))]
        elif different=='frist':
            grp_labels = [grp_labels[0]]
        elif different=='last':
            grp_labels = [grp_labels[-1]]
        elif different=='drop':
            grp_labels = list(set(grp_labels))
            if len(grp_labels) > 1:
                grp_labels = None
            # grp_labels = [grp_labels[0]] no need since len(grp_labels)==1
        return grp_labels
    result_df = data_df.groupby(key_columns).apply(_get_df_dict)
    result_df = result_df[~result_df.isna()]
    result_df = result_df.map(lambda x:x[0])
    return result_df.to_dict()
        
    result_df = data_df.groupby(key_columns).apply(_get_df_dict)
    return result_df.to_dict()

def dict_shrink(dict_value, interval=10):
    dict_sorted_value = sorted(dict_value.items(), key=lambda x:float(str(x[0]).split('-')[0]))
    dict_sorted_value = [dict_sorted_value[i:i+interval] for i in range(0, len(dict_sorted_value), interval)]
    dict_value = {f"{it[0][0].split('-')[0]}-{it[-1][0].split('-')[-1]}":sum([sub_it[1] for sub_it in it]) for it in dict_sorted_value}

    return dict_value