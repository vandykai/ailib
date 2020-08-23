from contextlib import contextmanager
import pandas as pd

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
