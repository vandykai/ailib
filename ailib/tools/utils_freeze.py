import logging
from collections.abc import Iterable

logger = logging.getLogger('__ailib__')

def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, module in model.named_modules():
        if name not in layer_names:
            continue
        for subname, param in module.named_parameters():
            param.requires_grad = not freeze
            logger.info(f"set requires_grad {name}.{subname} to {not freeze}")

def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)

def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)

def set_freeze_by_idxs(model, children_idxs, freeze=True):
    if not isinstance(children_idxs, Iterable):
        children_idxs = [children_idxs]
    num_child = len(list(model.children()))
    children_idxs = tuple(map(lambda idx: num_child + idx if idx < 0 else idx, children_idxs))
    for idx, (name, child) in enumerate(model.named_children()):
        if idx not in children_idxs:
            continue
        for subname, param in child.named_parameters():
            param.requires_grad = not freeze
            logger.info(f"set requires_grad {name}.{subname} to {not freeze}")
            
def freeze_by_idxs(model, children_idxs):
    set_freeze_by_idxs(model, children_idxs, True)

def unfreeze_by_idxs(model, children_idxs):
    set_freeze_by_idxs(model, children_idxs, False)