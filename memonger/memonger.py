import torch
import torch.nn as nn
from typing import overload
from .checkpoint import checkpoint_sequential
import math

__all__ = ['Sequential']

class Checkpoint():
    def __init__(self) -> None:
        raise NotImplementedError

class Sequential(nn.Sequential):
    @overload
    def __init__(self, arg: nn.Sequential) -> None:
        ...

    def __init__(self, *args):
        super(Sequential, self).__init__(*args)
        if len(args) == 1 and isinstance(args[0], nn.Sequential):
            super(Sequential, self).__init__()
            self._modules = args[0]._modules
        else:
            super(Sequential, self).__init__(*args)

    def forward(self, input):
        segments = int(math.sqrt(len(list(self.children()))))
        return checkpoint_sequential(functions=self, segments=segments, input=input)