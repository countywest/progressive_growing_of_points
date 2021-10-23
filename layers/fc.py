import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, ModuleList

class fc(nn.Module):
    def __init__(self, dims):
        super(fc, self).__init__()
        self.dims = dims
        fc_layers = []
        for i, dim in enumerate(self.dims):
            if i == len(self.dims) - 2:
                fc_layers.append(Linear(self.dims[i], self.dims[i + 1]))
                break
            else:
                fc_layers.append(Sequential(Linear(self.dims[i], self.dims[i + 1]), ReLU()))
        self.fc_layers = ModuleList(fc_layers)

    def forward(self, x):
        layer = x
        for fc in self.fc_layers:
            layer = fc(layer)
        return layer

class fcDecoder(fc):
    def __init__(self, dims):
        super(fcDecoder, self).__init__(dims)

    def forward(self, z):
        batch_size = z.shape[0]
        layer = z
        for fc in self.fc_layers:
            layer = fc(layer)
        pc = layer.view(batch_size, -1, 3)
        return pc