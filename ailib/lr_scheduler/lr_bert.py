from torch.optim.optimizer import Optimizer

class BertLR(object):
    '''
    Bert模型内定的学习率变化机制
    Example:
        >>> scheduler = BertLR(optimizer)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    '''
    def __init__(self, optimizer, learning_rate, t_total, warmup):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.t_total = t_total
        self.warmup = warmup

    # 线性预热方式
    def warmup_linear(self, x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x

    def batch_step(self, metrics, training_step=None):
        lr_this_step = self.learning_rate * self.warmup_linear(training_step / self.t_total,self.warmup)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr_this_step