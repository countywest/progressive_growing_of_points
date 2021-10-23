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
from torch.nn import Linear, ModuleList
import pdb
tree_arch = {}
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [4, 2, 4, 4, 4, 4]
tree_arch[8] = [4, 2, 2, 2, 2, 2, 4, 4]
tree_arch[10] = [4, 2, 2, 2, 2, 2, 2, 2, 2, 2]

class MultiResBlock1d(nn.Module):
    def __init__(self, name, in_channels, out_channels, up_ratio):
        super(MultiResBlock1d, self).__init__()
        self.name = name
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_ratio = up_ratio
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        

        self.conv0 = nn.Sequential()
        self.conv0.add_module('{}_conv0'.format(self.name),
                nn.ConvTranspose1d(self.in_channels*2, 
                                   self.out_channels, 
                                   kernel_size=self.up_ratio, 
                                   stride=self.up_ratio, 
                                   padding=0))
        self.conv0.add_module('{}_activation0'.format(self.name), nn.ReLU(inplace=True))

        self.conv1 = nn.Sequential()
        self.conv1.add_module('{}_conv1'.format(self.name),
                nn.ConvTranspose1d(self.in_channels*3, 
                                   self.out_channels, 
                                   kernel_size=self.up_ratio, 
                                   stride=self.up_ratio, 
                                   padding=0))
        self.conv1.add_module('{}_activation1'.format(self.name), nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential()
        self.conv2.add_module('{}_conv2'.format(self.name),
                nn.ConvTranspose1d(self.in_channels*2, 
                                   self.out_channels, 
                                   kernel_size=self.up_ratio, 
                                   stride=self.up_ratio, 
                                   padding=0))
        self.conv2.add_module('{}_activation2'.format(self.name), nn.ReLU(inplace=True))

    def forward(self, x):
        x0 = x[0]
        x1 = x[1]
        x2 = x[2]

        in0 = torch.cat((x0, self.upsample(x1)), 1)
        in1 = torch.cat((self.pool(x0), x1, self.upsample(x2)), 1)
        in2 = torch.cat((self.pool(x1), x2), 1)

        out0 = self.conv0(in0)
        out1 = self.conv1(in1)
        out2 = self.conv2(in2)

        return [out0, out1, out2]


class MRTDecoder(nn.Module):
    def __init__(self, z_dim, nlevels, feat_dims, num_output_points):
        super(MRTDecoder, self).__init__()
        self.z_dim = z_dim
        self.nlevels = nlevels
        self.feat_dims = feat_dims
        self.num_output_points = num_output_points
        self.tarch, self.num_nodes = self.get_arch()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.base_size = int(self.tarch[0])
        self.fc1 = Linear(self.z_dim, self.base_size * self.feat_dims[0])
        
        upconv_list = []
        for level in range(1, self.nlevels):
            upconv_list.append(self.upconv(level))
        self.upconv_list = ModuleList(upconv_list)

        self.final_conv = nn.Sequential()
        self.final_conv.add_module('final_conv1',
                nn.ConvTranspose1d(self.feat_dims[-1]*3, 32, kernel_size=1, stride=1, padding=0))
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
        return MultiResBlock1d('up_%d' % level, in_channels, out_channels, up_ratio)

    def concat_multires(self, node):
        node_0 = node[0]
        node_1 = self.upsample(node[1])
        node_2 = self.upsample(self.upsample(node[2]))
        node_concat = torch.cat((node_0, node_1, node_2), 1)
        return node_concat

    def forward(self, z):
        batch_size = z.shape[0]
        node_0 = self.fc1(z).view(batch_size, -1, self.base_size)       
        node_1 = self.pool(node_0)
        node_2 = self.pool(node_1)
        node = [node_0, node_1, node_2]

        for upconv in self.upconv_list:
            node = upconv(node)

        node_concat = self.concat_multires(node)
        out = self.final_conv(node_concat)
        out = torch.transpose(out, 1, 2).contiguous()
        return out