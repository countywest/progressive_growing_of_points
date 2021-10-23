''' 
Code from
https://github.com/lynetcha/completion3d/blob/master/tensorflow/models/TopNet.py

revised by Hyeontae Son
'''
import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import Sequential, Linear, ReLU, Tanh, ModuleList

tree_arch = {}
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]
tree_arch[11] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

class topnet(nn.Module):
    def __init__(self, z_dim, nlevels, node_feat_dim, num_output_points):
        super(topnet, self).__init__()
        self.z_dim = z_dim
        self.nlevels = nlevels
        self.node_feat_dim = node_feat_dim
        self.num_output_points = num_output_points
        self.tarch, self.num_nodes = self.get_arch()


        # for level0
        self.fc1 = Sequential(Linear(self.z_dim, 256), ReLU(),
                              Linear(256, 64), ReLU(),
                              Linear(64, self.node_feat_dim * int(self.tarch[0])), Tanh()
                              )

        self.input_code_dim = self.z_dim + self.node_feat_dim

        # for level1 ~ level(nlevels-1)
        upconv_list = []
        for level in range(1, self.nlevels):
            upconv_list.append(self.upconv(level))
        self.upconv_list = ModuleList(upconv_list)

        # make leaf node(level(nlevels-1)) to point cloud
        self.last_mlp = Sequential(Linear(self.node_feat_dim, 3), Tanh())

    def upconv(self, level):
        input_code_dim = self.input_code_dim
        output_code_dim = self.node_feat_dim
        up_ratio = self.tarch[level]
        return Sequential(Linear(input_code_dim, int(input_code_dim / 2)), ReLU(),
                          Linear(int(input_code_dim / 2), int(input_code_dim / 4)), ReLU(),
                          Linear(int(input_code_dim / 4), int(input_code_dim / 8)), ReLU(),
                          Linear(int(input_code_dim / 8), output_code_dim * up_ratio), Tanh()
                         )

    def get_arch(self):
        logmult = int(math.log2(self.num_output_points / 2048))
        tarch = tree_arch[self.nlevels]
        if self.num_output_points == 16384:
            while logmult > 0:
                last_min_pos = np.where(tarch == np.min(tarch))[0][-1]
                tarch[last_min_pos] *= 2
                logmult -= 1

        # number of node for each level
        num_nodes = []
        for i, up_ratio in enumerate(tarch):
            if i == 0:
                num_nodes.append(up_ratio)
            else:
                last_num_node = num_nodes[-1]
                num_nodes.append(up_ratio * last_num_node)

        return tarch, num_nodes

    def forward(self, z):
        batch_size = z.shape[0]
        node = self.fc1(z)
        node = node.view(-1, int(self.tarch[0]), self.node_feat_dim)

        for upconv in self.upconv_list:
            z_tiled = torch.unsqueeze(z, 1).repeat(1, node.shape[1], 1)
            node = torch.cat((z_tiled, node), dim=2)
            node = upconv(node)
            node = node.view(batch_size, -1, self.node_feat_dim)

        pc = self.last_mlp(node).contiguous()
        return pc
