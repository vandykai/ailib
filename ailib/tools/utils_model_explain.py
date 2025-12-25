
import torch
import torch.nn as nn

class Explain(nn.Module):

    def __init__(self, model):
        super().__init__()
        self._model = model

    def forward(self, inputs):
        return self._model(inputs)

    def explain(self, inputs):
        explain = {"w":None, "b":None}
        requires_grad = inputs.requires_grad
        training = self._model.training
        inputs.requires_grad = True
        self._model.zero_grad()
        out = self._model(inputs)
        out.backward(torch.ones_like(out))
        explain['w'] = inputs.grad.cpu().detach()
        explain['b'] = (out - inputs.grad.mm(inputs.T)).cpu().detach()
        inputs.requires_grad = requires_grad
        self._model.train(training)
        return out, explain