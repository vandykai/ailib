
from .stateful_unit import StatefulUnit
import torch

class Normalization(StatefulUnit):
    '''
    computes the feature average and current value
    '''

    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.mean = torch.zeros([input_dim])
        self.var = torch.zeros([input_dim])

        self.sum = torch.zeros([input_dim])
        self.square_sum = torch.zeros([input_dim])
        self.count = 0

    def fit(self, data_iter):
        for step, batch in enumerate(data_iter):
            inputs, _ = batch
            inputs = {'feature':torch.sparse_coo_tensor(inputs['i'], inputs['v'], [inputs['batch_size'], self.input_dim])}
            self.sum += torch.sparse.sum(inputs['feature'], dim=0)
            self.square_sum += torch.sparse.sum(inputs['feature'].pow(2), dim=0)
            self.count += len(inputs['feature'])
        self.mean = self.sum / self.count
        self.var = self.square_sum / (self.count-1) - self.sum.pow(2)/(self.count * (self.count-1))
            
    def transform(self, inputs):
        pass
        #inputs['feature'] = feature-


    def update(self, key, val, n=1):
        self.vals[key].extend([val]*n)
        self.sum[key] += val * n
        self.count[key] += n
        self.avg[key] = self.sum[key] / self.count[key]