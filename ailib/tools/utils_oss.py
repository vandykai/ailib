import gzip
import logging
import os
from pathlib import Path

import oss2
import pandas as pd
from tqdm.auto import tqdm

logger = logging.getLogger('__ailib__')

def get_oss_files(oss_dir, oss_config):
    file_paths = []
    oss_dir = Path(oss_dir)
    auth = oss2.Auth(oss_config['accessKeyID'], oss_config['accessKeySecret'])
    bucket_name = oss_dir.parts[1]
    bucket = oss2.Bucket(auth, oss_config['endpoint'], bucket_name)
    for obj in oss2.ObjectIterator(bucket, prefix=os.sep.join(oss_dir.parts[2:])):
        file_paths.append(os.sep.join([oss_dir.parts[0], oss_dir.parts[1], obj.key]))
    return file_paths

def get_oss_open_files(oss_dir, oss_config):
    file_paths = []
    oss_dir = Path(oss_dir)
    auth = oss2.Auth(oss_config['accessKeyID'], oss_config['accessKeySecret'])
    bucket_name = oss_dir.parts[1]
    bucket = oss2.Bucket(auth, oss_config['endpoint'], bucket_name)
    for obj in oss2.ObjectIterator(bucket, prefix=os.sep.join(oss_dir.parts[2:])):
        if obj.key.endswith('.gz'):
            file_paths.append(gzip.open(bucket.get_object(obj.key)))
        else:
            file_paths.append(bucket.get_object(obj.key))
    return file_paths

def open_oss_file(oss_path, oss_config):
    oss_path = Path(oss_path)
    auth = oss2.Auth(oss_config['accessKeyID'], oss_config['accessKeySecret'])
    bucket_name = oss_path.parts[1]
    bucket = oss2.Bucket(auth, oss_config['endpoint'], bucket_name)
    if str(oss_path).endswith('.gz'):
        return gzip.open(bucket.get_object(os.sep.join(oss_path.parts[2:])))
    else:
        return bucket.get_object(os.sep.join(oss_path.parts[2:]))

def load_oss_fold_data(oss_dir, oss_config, func=pd.read_csv, **kwargs):
    datas = []
    oss_dir = Path(oss_dir)
    auth = oss2.Auth(oss_config['accessKeyID'], oss_config['accessKeySecret'])
    bucket_name = oss_dir.parts[1]
    bucket = oss2.Bucket(auth, oss_config['endpoint'], bucket_name)
    for obj in oss2.ObjectIterator(bucket, prefix=os.sep.join(oss_dir.parts[2:])):
        if obj.key.endswith('.gz'):
            datas.append(func(gzip.open(bucket.get_object(obj.key)), **kwargs))
        else:
            datas.append(func(bucket.get_object(obj.key), **kwargs))
    return pd.concat(datas, ignore_index = True)

def load_oss_files(oss_paths, oss_config, func=pd.read_csv, **kwargs):
    datas = []
    auth = oss2.Auth(oss_config['accessKeyID'], oss_config['accessKeySecret'])
    for oss_path in oss_paths:
        oss_path = Path(oss_path)
        bucket_name = oss_path.parts[1]
        bucket = oss2.Bucket(auth, oss_config['endpoint'], bucket_name)
        oss_path = os.sep.join(oss_path.parts[2:])
        if oss_path.endswith('.gz'):
            datas.append(func(gzip.open(bucket.get_object(oss_path)), **kwargs))
        else:
            datas.append(func(bucket.get_object(oss_path), **kwargs))
    return pd.concat(datas, ignore_index = True)

def get_oss_files_size(oss_paths, oss_config):
    total_content_length = 0
    auth = oss2.Auth(oss_config['accessKeyID'], oss_config['accessKeySecret'])
    for oss_path in oss_paths:
        oss_path = Path(oss_path)
        bucket_name = oss_path.parts[1]
        bucket = oss2.Bucket(auth, oss_config['endpoint'], bucket_name)
        meta_info = bucket.get_object_meta(os.sep.join(oss_path.parts[2:]))
        total_content_length += meta_info.content_length
    return total_content_length

def upload_file_to_oss(local_file, oss_path, oss_config):
    oss_path = Path(oss_path)
    auth = oss2.Auth(oss_config['accessKeyID'], oss_config['accessKeySecret'])
    bucket_name = oss_path.parts[1]
    bucket = oss2.Bucket(auth, oss_config['endpoint'], bucket_name)
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

def upload_fold_to_oss(local_dir, pattern, oss_dir, oss_config):
    oss_dir = Path(oss_dir)
    local_dir = Path(local_dir)
    auth = oss2.Auth(oss_config['accessKeyID'], oss_config['accessKeySecret'])
    bucket_name = oss_dir.parts[1]
    bucket = oss2.Bucket(auth, oss_config['endpoint'], bucket_name)
    for it in local_dir.glob(pattern):
        retry = 3
        while retry > 0:
            oss_path = os.sep.join(oss_dir.parts[2:]+it.relative_to(local_dir).parts)
            logger.info(f'uploading:{it} to {oss_path}')
            result = bucket.put_object_from_file(oss_path, it)
            if result.status == 200:
                break
            retry -= 1
            logger.error(f'retry:{retry} upload:{it}')