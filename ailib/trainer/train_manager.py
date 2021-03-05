import os
import time
import logging
from itertools import count
from pathlib import Path
from ailib.tools.utils_persistence import save_model, load_model, save_dill

logger = logging.getLogger()

class TrainConfig():
    def __init__(self, model_config):
        self.model_config = model_config
        self.device = "cuda:0"
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_func = None
        self.target_field = None
        self.input_field = None

        self.max_epoch = None
        self.output_dir = Path("./outputs")/model_config.model_name/time.strftime("%Y-%m-%d-%H-%M",time.localtime(time.time()))
        self.max_grad_norm = None # eg 5.0
        self.early_stop_epoch = 10
        self.is_valid_val_better = lambda val, best_val:val>best_val
        self.save_last_checkpoint = True

class TrainState():
    def __init__(self):
        self.stop_train = False
        self.score_improved = False

class TrainManager():

    def __init__(self, train_config):
        self.config = train_config
        self.reset()

    def source_config(self):
        if self.config.model:
            self.config.model.to(self.config.device)
        if not self.config.output_dir.exists():
            os.makedirs(self.config.output_dir)

        if self.config.max_epoch is None:
            self.epoch_iter = count(self.current_epoch)
        else:
            self.epoch_iter = range(self.current_epoch, 1+self.config.max_epoch)
        self.best_model_path = self.config.output_dir/'best-model.bin'
        self.last_model_path = self.config.output_dir/'last-model.bin'
        self.last_checkpoint_path =  self.config.output_dir/'last_checkpoint.dill'

    def reset(self):
        self.train_vals = []
        self.valid_vals = []
        self.early_stop_tolerance = 0
        self.best_valid_val = None
        self.state = TrainState()
        self.current_epoch = 1
        if self.config.scheduler:
            self.config.scheduler.reset()
        self.source_config()

    def epoch_step(self, train_val, valid_val, epoch, **kwargs):
        self.current_epoch = epoch
        self.train_vals.append((train_val, epoch))
        self.valid_vals.append((valid_val, epoch))
        self.state.score_improved = False
        if (not self.best_valid_val) or self.config.is_valid_val_better(valid_val, self.best_valid_val):
            logger.info(f"Epoch {epoch}: best_valid_val improved from {self.best_valid_val} to {valid_val}")
            self.early_stop_tolerance = 0
            self.best_valid_val = valid_val
            self.state.score_improved = True
        else:
            self.early_stop_tolerance += 1

        if self.state.score_improved:
            save_model(self.config.model, self.best_model_path, epoch=epoch, **kwargs)
            logger.info("saved best model to disk.")

        if self.early_stop_tolerance > self.config.early_stop_epoch or epoch == self.config.max_epoch:
            self.state.stop_train = True
            save_model(self.config.model, self.last_model_path, epoch=epoch, **kwargs)
            logger.info("stop train and saved best model to disk.")
        if self.config.save_last_checkpoint:
            save_dill(self, self.last_checkpoint_path)
