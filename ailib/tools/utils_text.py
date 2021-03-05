import re

def reg_find_all(reg_list, text, handler=None):
    key_word = []
    spans = []
    for reg in reg_list:
        if isinstance(reg, str):
            reg = re.compile(reg, re.M|re.I)
        match_objs = reg.finditer(text)
        for match_obj in match_objs:
            groups = match_obj.groups()
            actural_start = next((i for i in range(len(groups)) if groups[i] is not None), 0)
            actural_end = next((i for i in range(len(groups)-1, 0, -1) if groups[i] is not None), 0)
            actural_start = match_obj.start(actural_start + 1)
            actural_end = match_obj.end(actural_end + 1)
            spans.append((actural_start, actural_end))
            groups = filter(lambda x: x, groups)
            key_word.append("".join(groups) if handler is None else handler(groups))
    return key_word, spans

def strip(chars, text):
    """去除首尾的字符
    :type text: string
    :type chars: string
    :rtype: string
    """
    if chars is None:
        reg = re.compile('^ *| *$')
    else:
        reg = re.compile(r'^[' + chars + ']*|[' + chars + ']*$')
    return reg.sub('', text)

def lstrip(chars, text):
    """去除首部的字符
    :type text: string
    :type chars: string
    :rtype: string
    """
    if chars is None:
        reg = re.compile('^ *| *$')
    else:
        reg = re.compile(r'^[' + chars + ']*')
    return reg.sub('', text)

def rstrip(chars, text):
    """去除尾部的字符
    :type text: string
    :type chars: string
    :rtype: string
    """
    if chars is None:
        reg = re.compile('^ *| *$')
    else:
        reg = re.compile(r'[' + chars + ']*$')
    return reg.sub('', text)

# def replace_dict(text, replace_dict_, *re_args):
#     """替换文本中包含的在字典中的key为value
#     :type text: string
#     :type replace_dict_: dict
#     :type re_args: regex flags
#     :rtype: string
#     """
#     def replace_dict_func(match_obj):
#         key = "".join(match_obj.groups())
#         return replace_dict_.get(key, key)
#     keys = sorted(replace_dict_.keys(), key=len, reverse=True)
#     reg = re.compile(f"({'|'.join([re.escape(key) for key in keys])})", *re_args)
#     return reg.sub(replace_dict_func, text)

def replace_dict(text, replace_dict_, escape, *re_args):
    """替换文本中包含的在字典中的key为value
    :type text: string
    :type replace_dict_: dict
    :type escape: replace_dict_中key值是否转义
    :type re_args: regex flags
    :rtype: string
    """
    keys = sorted(replace_dict_.keys(), key=len, reverse=True)
    def replace_dict_func(match_obj):
        groups = match_obj.groups()
        for index in range(len(groups)):
            if groups[index]:
                value = replace_dict_.get(keys[index])
                return value
        return keys
    if escape:
        reg = re.compile(f"({'|'.join([re.escape(key) for key in keys])})", *re_args)
    else:
        reg = re.compile(f"({')|('.join(keys)})", *re_args)
    return reg.sub(replace_dict_func, text)
