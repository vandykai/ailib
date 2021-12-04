from pathlib import Path
import logging
import torch.nn.init as init
import torch.nn as nn

LOG_LEVEL_MAP = {
    'NOTSET':logging.NOTSET, # 0
    "DEBUG":logging.DEBUG, #10
    "INFO":logging.INFO, #20
    "WARNING":logging.WARNING, #30
    "ERROR":logging.ERROR, #40
    "CRITICAL":logging.CRITICAL, #50
}

def init_logger(log_level=logging.INFO, log_file=None, logger_name='__ailib__'):
    '''
    Example:
        >>> init_logger(log_file=log_file)
        >>> logger.info("abc'")
    '''
    if isinstance(log_file, Path):
        log_file = str(log_file)
    if type(log_level) == str:
        log_level = LOG_LEVEL_MAP[log_level.upper()]

    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                                   datefmt='%m/%d/%Y %H:%M:%S')
    handlers = []
    logger = logging.getLogger(logger_name)
    logger.propagate= False
    logger.setLevel(log_level)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    handlers.append(console_handler)
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        # file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        handlers.append(file_handler)
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
