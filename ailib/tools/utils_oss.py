from functools import partial
import gzip
import logging
import os
from pathlib import Path

import oss2
import pandas as pd
from tqdm.auto import tqdm
from ailib.tools.utils_file import maybe_download, temporary_path
import tempfile
import configparser
import os



logger = logging.getLogger('__ailib__')


def get_oss_bucket(oss_path):
    config = configparser.ConfigParser()
    config.read(f"{os.path.expanduser('~')}/.ossutilconfig")
    bucket_name = oss_path.parts[1]
    auth = oss2.Auth(config['Credentials']['accessKeyID'], config['Credentials']['accessKeySecret'])
    endpoint = config["Bucket-Endpoint"][bucket_name] if ("Bucket-Endpoint" in config and bucket_name in config["Bucket-Endpoint"]) else config['endpoint']
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    return bucket

def get_oss_files(oss_dir):
    file_paths = []
    oss_dir = Path(oss_dir)
    bucket = get_oss_bucket(oss_dir)
    for obj in oss2.ObjectIterator(bucket, prefix=os.sep.join(oss_dir.parts[2:])):
        file_paths.append(os.sep.join([oss_dir.parts[0], oss_dir.parts[1], obj.key]))
    return file_paths

def get_oss_open_files(oss_dir):
    file_paths = []
    oss_dir = Path(oss_dir)
    bucket = get_oss_bucket(oss_dir)
    for obj in oss2.ObjectIterator(bucket, prefix=os.sep.join(oss_dir.parts[2:])):
        if obj.key.endswith('.gz'):
            file_paths.append(gzip.open(bucket.get_object(obj.key)))
        else:
            file_paths.append(bucket.get_object(obj.key))
    return file_paths

def open_oss_file(oss_path):
    oss_path = Path(oss_path)
    bucket = get_oss_bucket(oss_path)
    if str(oss_path).endswith('.gz'):
        return gzip.open(bucket.get_object(os.sep.join(oss_path.parts[2:])))
    else:
        return bucket.get_object(os.sep.join(oss_path.parts[2:]))

def oss_file_auto_reader(oss_path, bucket, **kwargs):
    oss_path = Path(oss_path)
    file_type_route = {
        'xlsx':('.xlsx', '.xlsx.gz'),
        'csv': ('.csv', '.csv.gz', '.tsv', '.tsv.gz')
    }
    suffix = ''.join(oss_path.suffixes)
    if suffix in file_type_route['xlsx']:
        oss_path = {"oss_path":oss_path, "oss_file":get_oss_download_url(oss_path)}
        return read_oss_excel(oss_path, **kwargs)
    elif suffix in file_type_route['csv']:
        oss_path = {"oss_path":oss_path, "oss_file":bucket.get_object(os.sep.join(oss_path.parts[2:]))}
        return read_oss_csv(oss_path, **kwargs)
    else:
        raise ValueError(f"{oss_path} file type not in {set(file_type_route.values())}")

def load_oss_fold_data(oss_dir, func=oss_file_auto_reader, **kwargs):
    datas = []
    oss_dir = Path(oss_dir)
    bucket = get_oss_bucket(oss_dir)
    oss_dir_head = Path(os.sep.join(oss_dir.parts[:2]))
    if func.__name__ == oss_file_auto_reader.__name__:
        func = partial(func, bucket=bucket)
    for obj in oss2.ObjectIterator(bucket, prefix=os.sep.join(oss_dir.parts[2:])):
        if obj.size!=0:
            df = func(oss_dir_head/obj.key, **kwargs)
            datas.append(df)
    return pd.concat(datas, ignore_index = True)

def load_oss_fold_data_dict(oss_dir, func=oss_file_auto_reader, **kwargs):
    datas = {}
    oss_dir = Path(oss_dir)
    bucket = get_oss_bucket(oss_dir)
    oss_dir_head = Path(os.sep.join(oss_dir.parts[:2]))
    if func.__name__ == oss_file_auto_reader.__name__:
        func = partial(func, bucket=bucket)
    for obj in oss2.ObjectIterator(bucket, prefix=os.sep.join(oss_dir.parts[2:])):
        if obj.size!=0:
            datas[obj.key] = func(oss_dir_head/obj.key, **kwargs)
    return datas

def load_oss_files(oss_paths, func=oss_file_auto_reader, **kwargs):
    datas = []
    func_partial = func
    for oss_path in oss_paths:
        oss_path = Path(oss_path)
        bucket = get_oss_bucket(oss_path)
        if func.__name__ == oss_file_auto_reader.__name__:
            func_partial = partial(func, bucket=bucket)
        datas.append(func_partial(oss_path, **kwargs))
    return pd.concat(datas, ignore_index = True)

def get_oss_files_size(oss_paths):
    total_content_length = 0
    for oss_path in oss_paths:
        oss_path = Path(oss_path)
        bucket = get_oss_bucket(oss_path)
        meta_info = bucket.get_object_meta(os.sep.join(oss_path.parts[2:]))
        total_content_length += meta_info.content_length
    return total_content_length

def upload_file_to_oss(local_file, oss_path):
    oss_path = Path(oss_path)
    bucket = get_oss_bucket(oss_path)
    oss_path = os.sep.join(oss_path.parts[2:])
    retry = 3
    while retry > 0:
        logger.info(f'uploading:{local_file} to {oss_path}')
        result = bucket.put_object_from_file(oss_path, local_file)
        if result.status == 200:
            return 1
        retry -= 1
        logger.error(f'retry:{retry} upload:{local_file}')
    return 0

def upload_fold_to_oss(local_dir, pattern, oss_dir):
    oss_dir = Path(oss_dir)
    local_dir = Path(local_dir)
    bucket = get_oss_bucket(oss_dir)
    for it in local_dir.rglob(pattern):
        if not it.is_file():
            continue
        retry = 3
        while retry > 0:
            oss_path = os.sep.join(oss_dir.parts[2:]+it.relative_to(local_dir).parts)
            logger.info(f'uploading:{it} to {oss_path}')
            result = bucket.put_object_from_file(oss_path, it)
            if result.status == 200:
                break
            retry -= 1
            logger.error(f'retry:{retry} upload:{it}')


def get_oss_upload_urls(oss_dir, files, expires=48*60*60):
    file_urls = {}
    oss_dir = Path(oss_dir)
    bucket = get_oss_bucket(oss_dir)
    headers = {}
    for file in files:
        file_path = oss_dir/file
        url = bucket.sign_url('PUT', os.sep.join(file_path.parts[2:]), expires, slash_safe=True, headers=headers)
        file_urls[str(file_path)] = url
    return file_urls

def get_oss_download_urls(oss_dir, expires=48*60*60):
    file_urls = {}
    oss_dir = Path(oss_dir)
    bucket = get_oss_bucket(oss_dir)
    oss_dir_head = Path(os.sep.join(oss_dir.parts[:2]))
    headers = {}
    #headers['Accept-Encoding'] = 'gzip'
    params = dict()
    for obj in oss2.ObjectIterator(bucket, prefix=os.sep.join(oss_dir.parts[2:])):
        url = bucket.sign_url('GET', obj.key, expires, slash_safe=True, headers=headers, params=params)
        oss_path = oss_dir_head/Path(obj.key)
        file_urls[str(oss_path)] = url
    return file_urls

def get_oss_download_url(oss_path, expires=48*60*60):
    oss_path = Path(oss_path)
    bucket = get_oss_bucket(oss_path)
    headers = {}
    #headers['Accept-Encoding'] = 'gzip'
    params = dict()
    url = bucket.sign_url('GET', os.sep.join(oss_path.parts[2:]), expires, slash_safe=True, headers=headers, params=params)
    return url

def read_oss_csv(oss_path, **kwargs):
    if str(oss_path['oss_path']).endswith('gz'):
        with gzip.open(oss_path['oss_file']) as file:
            return pd.read_csv(file, **kwargs)
    else:
        return pd.read_csv(oss_path['oss_file'], **kwargs)

def read_oss_excel(oss_path, **kwargs):
    with tempfile.TemporaryDirectory() as path:
        filepath = maybe_download(url=oss_path['oss_file'], work_directory=path)
        if str(oss_path['oss_path']).endswith('gz'):
             with gzip.open(filepath) as file:
                return pd.read_excel(file, **kwargs)
        else:
            return pd.read_excel(filepath, **kwargs)
    