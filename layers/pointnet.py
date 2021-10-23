import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, ModuleList
from layers.fc import fc

class pointnet(nn.Module):
    def __init__(self, feat_dims_list):
        super(pointnet, self).__init__()
        self.mlp_dims = feat_dims_list
        shared_mlp_list = []
        for i, mlp in enumerate(self.mlp_dims):
            if i == len(self.mlp_dims) - 2:
                shared_mlp_list.append(Linear(self.mlp_dims[i], self.mlp_dims[i + 1]))
                break
            else:
                shared_mlp_list.append(Sequential(Linear(self.mlp_dims[i], self.mlp_dims[i + 1]), ReLU()))

        self.shared_mlps = ModuleList(shared_mlp_list)

    def forward(self, pc):
        x = pc
        for shared_mlp in self.shared_mlps:
            x = shared_mlp(x)
        x = torch.max(x, 1)[0].view(-1, self.mlp_dims[-1])
        return x

class pointnetDiscriminator(nn.Module):
    def __init__(self, mlp_dims, fc_dims):
        super(pointnetDiscriminator, self).__init__()
        self.mlp_dims = mlp_dims
        self.fc_dims = fc_dims
        self.pointnet = pointnet(self.mlp_dims)
        self.to_logit = fc(self.fc_dims)

    def forward(self, pc):
        gfv = self.pointnet(pc)
        logit = self.to_logit(gfv)
        return logit
