from torch.optim.optimizer import Optimizer

class NoamLR(object):
    '''
    主要参考论文<< Attention Is All You Need>>中的学习更新方式
    Example:
        >>> scheduler = NoamLR(d_model,factor,warm_up,optimizer)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         glopab_step += 1
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step(global_step)
        >>>     validate(...)
    '''
    def __init__(self,d_model,factor,warm_up,optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        self.warm_up = warm_up
        self.factor = factor
        self.d_model = d_model
        self._lr = 0

    def get_lr(self,step):
        lr = self.factor * (self.d_model ** (-0.5) * min(step ** (-0.5),step * self.warm_up ** (-1.5)))
        return lr

    def batch_step(self,step):
        '''
        update parameters and rate
        :return:
        '''
        lr = self.get_lr(step)
        for p in self.optimizer.param_groups:
            p['lr'] = lr
        self._lr = lr