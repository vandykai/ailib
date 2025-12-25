import logging
import sys
from pathlib import Path

import torch.nn as nn
import torch.nn.init as init

LOG_LEVEL_MAP = {
    'NOTSET':logging.NOTSET, # 0
    "DEBUG":logging.DEBUG, #10
    "INFO":logging.INFO, #20
    "WARNING":logging.WARNING, #30
    "ERROR":logging.ERROR, #40
    "CRITICAL":logging.CRITICAL, #50
}

class SingleLevelFilter(logging.Filter):
    def __init__(self, passlevel, reject):
        self.passlevel = passlevel
        self.reject = reject

    def filter(self, record): # return true mean pass
        if self.reject:
            return (record.levelno > self.passlevel)
        else:
            return (record.levelno <= self.passlevel)

class OSSLoggingHandler(logging.StreamHandler):
    def __init__(self, log_file, bucket):
        logging.StreamHandler.__init__(self)
        print(log_file)
        self._log_file = log_file
        self._bucket = bucket
        self._pos = self._bucket.append_object(self._log_file, 0, '')

    def emit(self, record):
        msg = self.format(record)
        msg += '\n'
        self._pos = self._bucket.append_object(self._log_file, self._pos.next_position, msg)


def init_logger(log_level=logging.INFO, log_file=None, bucket=None, logger_name='__ailib__'):
    '''
    Example:
        >>> init_logger(log_file=log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file, Path):
        log_file = str(log_file)
    if type(log_level) == str:
        log_level = LOG_LEVEL_MAP[log_level.upper()]

    log_format = logging.Formatter(fmt='[%(asctime)s] [%(name)s] [%(levelname)-8s] [%(process)-2d#%(threadName)-3s] ' +
                        '[%(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    handlers = []
    logger = logging.getLogger(logger_name)
    logger.propagate= False
    logger.setLevel(log_level)
    console_handler_stdout = logging.StreamHandler(sys.stdout)
    console_handler_stdout.setFormatter(log_format)
    stdout_filter = SingleLevelFilter(logging.INFO, False)
    console_handler_stdout.addFilter(stdout_filter)
    handlers.append(console_handler_stdout)

    console_handler_stderr = logging.StreamHandler(sys.stderr)
    console_handler_stderr.setFormatter(log_format)
    stderr_filter = SingleLevelFilter(logging.INFO, True)
    console_handler_stderr.addFilter(stderr_filter)
    handlers.append(console_handler_stderr)
    if log_file is not None:
        if bucket is not None:
            file_handler_stdout = OSSLoggingHandler(log_file+'.info', bucket)
            file_handler_stderr = OSSLoggingHandler(log_file+'.error', bucket)
        else:
            file_handler_stdout = logging.FileHandler(log_file+'.info')
            file_handler_stderr = logging.FileHandler(log_file+'.error')
        # file_handler.setLevel(log_file_level)
        file_handler_stdout.setFormatter(log_format)
        file_handler_stdout.addFilter(stdout_filter)
        file_handler_stderr.setFormatter(log_format)
        file_handler_stderr.addFilter(stderr_filter)
        handlers.append(file_handler_stdout)
        handlers.append(file_handler_stderr)
    logger.handlers = handlers

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)
