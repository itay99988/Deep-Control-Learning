import torch
from torch import nn
import torch.nn.functional as F


# This class inherits from nn.Linear (a linear layer of a neural network in PyTorch).
# The only modification that was needed in our case is to clone the weights of the linear layer after each use,
# so we can calculate gradients for the loss in case of lookahead > 0.
# (otherwise the gradients of the linear layer are overidden)
class MyLinear(nn.Linear):
    def forward(self, input):
        return F.linear(input, self.weight.clone(), self.bias.clone())
