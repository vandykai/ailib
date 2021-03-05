import torch
import numpy as np
import json
import pickle
import dill
import torch.nn as nn
from pathlib import Path
import logging

def save_pickle(data, file_path, **kwargs):
    '''
    保存成pickle文件
    :param data:
    :param file_name:
    :param pickle_path:
    :return:
    '''
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f, **kwargs)


def load_pickle(input_file, **kwargs):
    '''
    读取pickle文件
    :param pickle_path:
    :param file_name:
    :return:
    '''
    with open(str(input_file), 'rb') as f:
        data = pickle.load(f, **kwargs)
    return data

def save_dill(data, file_path):
    '''
    保存成dill文件
    :param data:
    :param file_name:
    :param dill_path:
    :return:
    '''
    if isinstance(file_path, Path):
        file_path = str(file_path)
    with open(file_path, 'wb') as f:
        dill.dump(data, f)


def load_dill(input_file):
    '''
    读取dill文件
    :param dill_path:
    :param file_name:
    :return:
    '''
    with open(str(input_file), 'rb') as f:
        data = dill.load(f)
    return data

def save_json(data, file_path):
    '''
    保存成json文件
    :param data:
    :param json_path:
    :param file_name:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    # if isinstance(data,dict):
    #     data = json.dumps(data)
    with open(str(file_path), 'w') as f:
        json.dump(data, f)

def save_numpy(data, file_path):
    '''
    保存成.npy文件
    :param data:
    :param file_path:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    np.save(str(file_path),data)

def load_numpy(file_path):
    '''
    加载.npy文件
    :param file_path:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    np.load(str(file_path))

def load_json(file_path):
    '''
    加载json文件
    :param json_path:
    :param file_name:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'r') as f:
        data = json.load(f)
    return data

def json_to_text(file_path,data):
    '''
    将json list写入text文件中
    :param file_path:
    :param data:
    :return:
    '''
    if not isinstance(file_path, Path):
        file_path = Path(file_path)
    with open(str(file_path), 'w') as fw:
        for line in data:
            line = json.dumps(line, ensure_ascii=False)
            fw.write(line + '\n')

def save_model(model, model_path, **kwargs):
    """ 存储不含有显卡信息的state_dict或model
    :param model:
    :param model_name:
    :param only_param:
    :param kwargs:other informations
    :return:
    """
    if isinstance(model_path, Path):
        model_path = str(model_path)
    if isinstance(model, nn.DataParallel):
        model = model.module
    state_dict = model.state_dict()
    for key in state_dict:
        state_dict[key] = state_dict[key].cpu()
    if kwargs:
        assert "state_dict" not in kwargs
        kwargs["state_dict"] = state_dict
    torch.save(kwargs, model_path)

def load_model(model, model_path, state_key="state_dict"):
    '''
    加载模型
    :param model:
    :param model_path:
    :param state_key:
    :return: model and other information dict
    '''
    if isinstance(model_path, Path):
        model_path = str(model_path)
    logging.info(f"loading model from {str(model_path)} .")
    states = torch.load(model_path)
    if state_key:
        state_dict = states[state_key]
        del states[state_key]
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(state_dict)
    else:
        model.load_state_dict(state_dict)
    return model, states

def restore_checkpoint(resume_path, model=None):
    '''
    加载模型
    :param resume_path:
    :param model:
    :param optimizer:
    :return:
    注意： 如果是加载Bert模型的话，需要调整，不能使用该模式
    可以使用模块自带的Bert_model.from_pretrained(state_dict = your save state_dict)
    '''
    if isinstance(resume_path, Path):
        resume_path = str(resume_path)
    checkpoint = torch.load(resume_path)
    best = checkpoint['best']
    start_epoch = checkpoint['epoch'] + 1
    states = checkpoint['state_dict']
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(states)
    else:
        model.load_state_dict(states)
    return [model,best,start_epoch]


def save_state_dict(obj, file_path):
    '''
    加载模型
    :param obj: 对象
    :param file_path: 状态保存的路径
    :return:
    '''
    pass

def load_state_dict(obj, file_path):
    """Loads the schedulers state.
    Arguments:
        state_dict (dict): scheduler state. Should be an object returned
            from a call to :meth:`state_dict`.
    """
    pass