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