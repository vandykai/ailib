from contextlib import contextmanager
import pandas as pd
import IPython

@contextmanager
def pd_display_all():
    max_columns = pd.options.display.max_columns
    max_rows = pd.options.display.max_rows
    pd.options.display.max_columns=None
    pd.options.display.max_rows=None
    yield None
    pd.options.display.max_columns = max_columns
    pd.options.display.max_rows = max_rows

@contextmanager
def pd_display_reset():
    max_columns = pd.options.display.max_columns
    max_rows = pd.options.display.max_rows
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
    yield None
    pd.options.display.max_columns = max_columns
    pd.options.display.max_rows = max_rows

def display_img(img_url):
    display(IPython.display.Image(img_url))

def display_html(html_str):
    display(IPython.display.HTML(html_str))

def display_pd(data_frame):
    with pd_display_all():
        display(data_frame)

import os


def is_jupyter():
    """Check if the module is running on Jupyter notebook/console.

    Returns:
        bool: True if the module is running on Jupyter notebook or Jupyter console,
        False otherwise.
    """
    try:
        shell_name = get_ipython().__class__.__name__
        if shell_name == "ZMQInteractiveShell":
            return True
        else:
            return False
    except NameError:
        return False


def is_databricks():
    """Check if the module is running on Databricks.

    Returns:
        bool: True if the module is running on Databricks notebook,
        False otherwise.
    """
    try:
        if os.path.realpath(".") == "/databricks/driver":
            return True
        else:
            return False
    except NameError:
        return False
