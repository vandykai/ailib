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