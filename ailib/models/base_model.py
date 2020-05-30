from torch import nn

class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = str(type(self))

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.state_dict(), path)

    def forward(self):
        raise NotImplementedError

    def init_weights(self):
        raise NotImplementedError

    def loss_function(self):
        raise NotImplementedError

    def optimizer(self):
        raise NotImplementedError


