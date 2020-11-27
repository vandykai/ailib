from collections import defaultdict

class RecoderMeter(object):
    '''
    computes and stores the average and current value
    Example:
        >>> loss = RecoderMeter()
        >>> for step,batch in enumerate(train_data):
        >>>     pred = self.model(batch)
        >>>     raw_loss = self.metrics(pred,target)
        >>>     loss.update("loss", raw_loss.item(),n = 1)
        >>> loss_avg = loss.avg["loss"]
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = defaultdict(list)
        self.avg = defaultdict(int)
        self.sum = defaultdict(int)
        self.count = defaultdict(int)

    def update(self, key, val, n=1):
        self.vals[key].extend([val]*n)
        self.sum[key] += val * n
        self.count[key] += n
        self.avg[key] = self.sum[key] / self.count[key]