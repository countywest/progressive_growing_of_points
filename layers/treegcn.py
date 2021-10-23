''' 
Code from
https://github.com/seowok/TreeGAN/blob/master/layers/gcn.py
https://github.com/seowok/TreeGAN/blob/master/model/gan_network.py

revised by Hyeontae Son
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import math

class TreeGCN(nn.Module):
    def __init__(self, depth, features, degrees, support=10, node=1, upsample=True, activation=True):
        self.depth = depth
        self.in_feature = features[depth]
        self.out_feature = features[depth + 1]
        self.node = node
        self.degree = degrees[depth]
        self.upsample = upsample
        self.activation = activation
        super(TreeGCN, self).__init__()

        self.W_root = nn.ModuleList(
            [nn.Linear(features[inx], self.out_feature, bias=False) for inx in range(self.depth + 1)])

        if self.upsample:
            self.W_branch = nn.Parameter(torch.FloatTensor(self.node, self.in_feature, self.degree * self.in_feature))

        self.W_loop = nn.Sequential(nn.Linear(self.in_feature, self.in_feature * support, bias=False),
                                    nn.Linear(self.in_feature * support, self.out_feature, bias=False))

        self.bias = nn.Parameter(torch.FloatTensor(1, self.degree, self.out_feature))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.init_param()

    def init_param(self):
        if self.upsample:
            init.xavier_uniform_(self.W_branch.data, gain=init.calculate_gain('relu'))

        stdv = 1. / math.sqrt(self.out_feature)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, tree):
        batch_size = tree[0].size(0)
        root = 0
        for inx in range(self.depth + 1):
            root_num = tree[inx].size(1)
            repeat_num = int(self.node / root_num)
            root_node = self.W_root[inx](tree[inx])
            root = root + root_node.repeat(1, 1, repeat_num).view(batch_size, -1, self.out_feature)

        branch = 0
        if self.upsample:
            branch = tree[-1].unsqueeze(2) @ self.W_branch
            branch = self.leaky_relu(branch)
            branch = branch.view(batch_size, self.node * self.degree, self.in_feature)

            branch = self.W_loop(branch)

            branch = root.repeat(1, 1, self.degree).view(batch_size, -1, self.out_feature) + branch
        else:
            branch = self.W_loop(tree[-1])

            branch = root + branch

        if self.activation:
            branch = self.leaky_relu(branch + self.bias.repeat(1, self.node, 1))
        tree.append(branch)

        return tree

class TreeGCNGenerator(nn.Module):
    def __init__(self, features, degrees, support):
        self.layer_num = len(features) - 1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(TreeGCNGenerator, self).__init__()

        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            if inx == self.layer_num - 1:
                self.gcn.add_module('TreeGCN_' + str(inx),
                                    TreeGCN(inx, features, degrees,
                                            support=support, node=vertex_num, upsample=True, activation=False))
            else:
                self.gcn.add_module('TreeGCN_' + str(inx),
                                    TreeGCN(inx, features, degrees,
                                            support=support, node=vertex_num, upsample=True, activation=True))
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, z):
        if len(z.shape) == 2: # (batch_size, noise_dim)
            z = z.unsqueeze(1) # (batch_size, 1, noise_dim)
        tree = [z]
        feat = self.gcn(tree)

        self.pointcloud = feat[-1]

        return self.pointcloud

    def getPointcloud(self):
        return self.pointcloud[-1]
