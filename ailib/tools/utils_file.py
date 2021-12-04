import json
import tempfile
from sklearn.datasets import load_svmlight_file
from pathlib import Path

def read_lines(file_path, *args):
    with open(file_path, *args) as f:
        lines = f.readlines()
    return lines

def file_de_duplication_line(file_name):
    all_line = []
    with open(file_name) as f:
        for line in f:
            line = line.strip()
            if line not in all_line:
                all_line.append(line)
    with open(file_name, "w") as f:
        for line in all_line:
            f.write(line+"\n")

def save_to_file(json_list, file_name):
    with open(file_name, "w") as f:
        for item in json_list:
            if type(item) != str:
                item = json.dumps(item, ensure_ascii=False)
            f.write(item+'\n')

def load_svmlight(X_iter, y_iter, svm_save_path=None, on_memory=False, **kwargs):
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
            X, y = load_svmlight_file(str(svm_save_path), **kwargs)
            return X, y
        fp = open(svm_save_path, 'wb+')
    for X_sample, y_sample in zip(X_iter, y_iter):
        fp.write((str(y_sample) + ' '+ X_sample + '\n').encode('utf-8'))
    fp.seek(0)
    X, y = load_svmlight_file(fp, **kwargs)
    fp.close()
    return X, y