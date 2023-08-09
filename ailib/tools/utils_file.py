import json
import logging
import math
import os
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path, PosixPath
from tempfile import TemporaryDirectory

import pandas as pd
import requests
from retrying import retry
from sklearn.datasets import load_svmlight_file
from tqdm import tqdm
from itertools import (takewhile, repeat)

logger = logging.getLogger('__ailib__')

def read_lines(file_path, *args):
    with open(file_path, *args) as f:
        lines = f.readlines()
    return lines

def file_de_duplication_line(file_path):
    all_line = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line not in all_line:
                all_line.append(line)
    with open(file_path, "w") as f:
        for line in all_line:
            f.write(line+"\n")

def save_to_file(json_list, file_path):
    with open(file_path, "w") as f:
        for item in json_list:
            if type(item) != str:
                item = json.dumps(item, ensure_ascii=False)
            f.write(item+'\n')

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def get_svmlight_dim(X_iter):
    h, w = 0, 0
    for X_sample in X_iter:
        h += 1
        w = max(w, int(X_sample.split(' ')[-1].split(':')[0]))
    return (h, w)

def load_svmlight(X_iter=None, y_iter=None, svm_save_path=None, on_memory=False, zero_based=True, **kwargs):
    """
    X_iter : Iterable
        svmlight format
    y_iter : Iterable
        target label
    n_features : int, default=None
        The number of features to use. If None, it will be inferred. This
        argument is useful to load several files that are subsets of a
        bigger sliced dataset: each subset might not have examples of
        every feature, hence the inferred shape might vary from one
        slice to another.
        n_features is only required if ``offset`` or ``length`` are passed a
        non-default value.

    dtype : numpy data type, default=np.float64
        Data type of dataset to be loaded. This will be the data type of the
        output numpy arrays ``X`` and ``y``.

    multilabel : bool, default=False
        Samples may have several labels each (see
        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html)

    zero_based : bool or "auto", default="auto"
        Whether column indices in f are zero-based (True) or one-based
        (False). If column indices are one-based, they are transformed to
        zero-based to match Python/NumPy conventions.
        If set to "auto", a heuristic check is applied to determine this from
        the file contents. Both kinds of files occur "in the wild", but they
        are unfortunately not self-identifying. Using "auto" or True should
        always be safe when no ``offset`` or ``length`` is passed.
        If ``offset`` or ``length`` are passed, the "auto" mode falls back
        to ``zero_based=True`` to avoid having the heuristic check yield
        inconsistent results on different segments of the file.

    query_id : bool, default=False
        If True, will return the query_id array for each file.

    offset : int, default=0
        Ignore the offset first bytes by seeking forward, then
        discarding the following bytes up until the next new line
        character.

    length : int, default=-1
        If strictly positive, stop reading any new line of data once the
        position in the file has reached the (offset + length) bytes threshold.

    Returns
    -------
    X : scipy.sparse matrix of shape (n_samples, n_features)

    y : ndarray of shape (n_samples,), or, in the multilabel a list of
        tuples of length n_samples.

    query_id : array of shape (n_samples,)
       query_id for each sample. Only returned when query_id is set to
       True.

    See Also
    --------
    load_svmlight_files : Similar function for loading multiple files in this
        format, enforcing the same number of features/columns on all of them.
    """
    if svm_save_path is None:
        if on_memory:
            fp = tempfile.TemporaryFile()
        else:
            fp = tempfile.NamedTemporaryFile()
    else:
        svm_save_path = Path(svm_save_path)
        if svm_save_path.exists():
            X, y = load_svmlight_file(str(svm_save_path), zero_based=zero_based, **kwargs)
            return X, y
        fp = open(svm_save_path, 'wb+')
    if y_iter is None:
        for X_sample in X_iter:
            if type(X_sample) == str:
                fp.write(('0 '+ X_sample + '\n').encode('utf-8'))
            else:
                fp.write(('0 '+ ' '.join(X_sample) + '\n').encode('utf-8'))
    else:
        for X_sample, y_sample in zip(X_iter, y_iter):
            if type(X_sample) == str:
                fp.write((str(y_sample) + ' '+ X_sample + '\n').encode('utf-8'))
            else:
                fp.write((str(y_sample) + ' '+ ' '.join(X_sample) + '\n').encode('utf-8'))
    fp.seek(0)
    X, y = load_svmlight_file(fp, zero_based=zero_based, **kwargs)
    fp.close()
    if y_iter is None:
        return X
    else:
        return X, y


def save_svmlight(X_iter=None, y_iter=None, svm_save_path=None):
    """
    X_iter : Iterable
        svmlight format
    y_iter : Iterable
        target label
    Returns
    -------
        svm_save_path：Path
    """
    svm_save_path = Path(svm_save_path)
    fp = open(svm_save_path, 'wb+')
    for X_sample, y_sample in zip(X_iter, y_iter):
        fp.write((str(y_sample) + ' '+ X_sample + '\n').encode('utf-8'))
    fp.close()
    return svm_save_path

def df_dict_to_excel(df_dict, file_name):
    writer = pd.ExcelWriter(file_name)
    for key, value in df_dict.items():
        value.to_excel(writer, key, index=False)
    writer.save()

def load_fold_data(fold, pattern='*', func=pd.read_csv, recursive=False, debug=False, **kwargs):
    datas = []
    if recursive:
        file_paths = Path(fold).rglob(pattern)
    else:
        file_paths = Path(fold).glob(pattern)
    for file_path in file_paths:
        if debug:
            print(file_path)
        try:
            datas.append(func(file_path, **kwargs))
        except pd.errors.EmptyDataError as e:
            logger.error(f"{file_path} is empty")
    return pd.concat(datas, ignore_index = True)

def get_files(fold, pattern='*', recursive=False):
    files = []
    if recursive:
        file_paths = Path(fold).rglob(pattern)
    else:
        file_paths = Path(fold).glob(pattern)
    return list(file_paths)

def load_files(file_paths, func=pd.read_csv, **kwargs):
    datas = []
    for file_path in file_paths:
        try:
            datas.append(func(file_path, **kwargs))
        except pd.errors.EmptyDataError as e:
            logger.error(f"{file_path} is empty")
    return pd.concat(datas, ignore_index = True)

def load_fold_data_iter(fold, pattern='*', func=pd.read_csv, recursive=False, split=None, debug=False, **kwargs):
    file_paths = []
    if recursive:
        file_paths = Path(fold).rglob(pattern)
    else:
        file_paths = Path(fold).glob(pattern)
    for file_path in file_paths:
        if debug:
            print(file_path)
        file_paths.append(file_path)

    step = math.ceil(len(file_paths)/split)
    for i in range(0, len(file_paths), step):
        datas = []
        for file_path in file_paths[i:i+step]:
            try:
                datas.append(func(file_path, **kwargs))
            except pd.errors.EmptyDataError as e:
                logger.error(f"{file_path} is empty")
        yield pd.concat(datas, ignore_index = True)

def split_file(file_path, partlines=0, header=True, names=None):
    if not isinstance(file_path, PosixPath):
        file_path = Path(file_path)
    if names and not names.endswith('\n'):
        names += '\n'
    current_part = 0
    with open(file_path, 'r') as fin:
        current_partlines = 0
        for idx, line in enumerate(fin):
            if idx == 0 and header:
                if names is None:
                    names = line
                continue
            if current_partlines % partlines == 0:
                fout = open(file_path.parent/(file_path.stem+f'_{current_part}'+ file_path.suffix), 'w')
                fout.write(names)
                current_part += 1
                current_partlines == 0
            fout.write(line)
            current_partlines += 1
        fout.close()

def split_fold_file(fold, pattern=None, partlines=0, header=True, names=None, out_file_name=None):
    if names and not names.endswith('\n'):
        names += '\n'
    current_part = 0
    current_partlines = 0
    fout = None
    for file_path in Path(fold).glob(pattern):
        with open(file_path, 'r') as fin:
            for idx, line in enumerate(fin):
                if idx == 0 and header:
                    if names is None:
                        names = line
                    continue
                if current_partlines % partlines == 0:
                    if fout is not None:
                        fout.close()
                    if out_file_name:
                        fout = open(out_file_name.format(current_part=current_part), 'w')
                    else:
                        fout = open(file_path.parent/(file_path.stem+f'_{current_part}'+ file_path.suffix), 'w')
                    fout.write(names)
                    current_part += 1
                fout.write(line)
                current_partlines += 1
    if fout is not None:
        fout.close()
    print(f"total {current_partlines} lines, {current_part} files")

def unzip_file(zip_src, dst_dir, clean_zip_file=False):
    """Unzip a file

    Args:
        zip_src (str): Zip file.
        dst_dir (str): Destination folder.
        clean_zip_file (bool): Whether or not to clean the zip file.
    """
    fz = zipfile.ZipFile(zip_src, "r")
    for file in fz.namelist():
        fz.extract(file, dst_dir)
    if clean_zip_file:
        os.remove(zip_src)

@retry(wait_random_min=1000, wait_random_max=5000, stop_max_attempt_number=5)
def maybe_download(url, filename=None, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.

    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.

    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            logger.info(f"Downloading {url}")
            total_size = int(r.headers.get("content-length", 0))
            block_size = 1024
            num_iterables = math.ceil(total_size / block_size)
            with open(filepath, "wb") as file:
                for data in tqdm(
                    r.iter_content(block_size),
                    total=num_iterables,
                    unit="KB",
                    unit_scale=True,
                ):
                    file.write(data)
        else:
            logger.error(f"Problem downloading {url}")
            r.raise_for_status()
    else:
        logger.info(f"File {filepath} already downloaded")
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError(f"Failed to verify {filepath}")

    return filepath

@contextmanager
def temporary_path(path=None):
    """Return a path to download data. If `path=None`, then it yields a temporal path that is eventually deleted,
    otherwise the real path of the input.

    Args:
        path (str): Path to download data.

    Returns:
        str: Real path where the data is stored.

    Examples:
        >>> with temporary_path() as path:
        >>> ... maybe_download(url="http://example.com/file.zip", work_directory=path)

    """
    if path is None:
        tmp_dir = TemporaryDirectory()
        try:
            yield tmp_dir.name
        finally:
            tmp_dir.cleanup()
    else:
        path = os.path.realpath(path)
        yield path

def get_files_size(paths):
    total_content_length = 0
    for path in paths:
        path = Path(path)
        total_content_length += path.stat().st_size
    return total_content_length

def count_line(file_name):
    buffer = 1024 * 1024
    with open(file_name) as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)