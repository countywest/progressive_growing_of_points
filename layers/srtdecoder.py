''' 
Code from
https://github.com/matheusgadelha/MRTNet/blob/master/models/AutoEncoder.py
https://github.com/matheusgadelha/MRTNet/blob/master/models/MRTDecoder.py

revised by Hyeontae Son
'''
import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import Sequential, Linear, ModuleList
tree_arch = {}
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]
tree_arch[11] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

class SRTDecoder(nn.Module):
    def __init__(self, z_dim, nlevels, feat_dims, num_output_points):
        super(SRTDecoder, self).__init__()
        self.z_dim = z_dim
        self.nlevels = nlevels
        self.feat_dims = feat_dims
        self.num_output_points = num_output_points
        self.tarch, self.num_nodes = self.get_arch()

        self.base_size = int(self.tarch[0])
        self.fc1 = Linear(self.z_dim, self.base_size * self.feat_dims[0])
        
        upconv_list = []
        for level in range(1, self.nlevels):
            upconv_list.append(self.upconv(level))
        self.upconv_list = ModuleList(upconv_list)

        self.final_conv = nn.Sequential()
        self.final_conv.add_module('final_conv1',
                nn.ConvTranspose1d(self.feat_dims[-1], 32, kernel_size=1, stride=1, padding=0))
        self.final_conv.add_module('relu_final',
                nn.ReLU(inplace=True))
        self.final_conv.add_module('final_conv2',
                nn.ConvTranspose1d(32, 3, kernel_size=1, stride=1, padding=0))
        self.final_conv.add_module('tanh_final',
                nn.Tanh())
    
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

    def upconv(self, level):
        in_channels = self.feat_dims[level-1]
        out_channels = self.feat_dims[level]
        up_ratio = self.tarch[level]
        return Sequential(
                          nn.ConvTranspose1d(in_channels, out_channels, kernel_size=up_ratio, stride=up_ratio, padding=0), 
                          nn.LeakyReLU(0.2, inplace=True)
                         )

    def forward(self, z):
        batch_size = z.shape[0]
        node = self.fc1(z).view(batch_size, -1, self.base_size)

        for upconv in self.upconv_list:
            node = upconv(node)

        out = self.final_conv(node)
        out = torch.transpose(out, 1, 2).contiguous()
        return out