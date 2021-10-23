''' 
Code from
https://github.com/matheusgadelha/MRTNet/blob/master/models/AutoEncoder.py
https://github.com/matheusgadelha/MRTNet/blob/master/models/MRTDecoder.py

revised by Hyeontae Son
'''
import torch
import torch.nn as nn
from torch.nn import Sequential, Tanh, Conv1d, ModuleList
from layers.srtdecoder import SRTDecoder
tree_arch = {}
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]
tree_arch[11] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

class SRTDecoderPG(SRTDecoder):
    def __init__(self, z_dim, nlevels, feat_dims, num_output_points):
        super(SRTDecoderPG, self).__init__(z_dim, nlevels, feat_dims, num_output_points)
        self.num_points_of_phase = self.num_nodes
        toPoint_list = []
        for intermediate_level in range(self.nlevels - 1):
            toPoint_list.append(Sequential(Conv1d(self.feat_dims[intermediate_level], 3, 1), Tanh()))

        self.toPoints = ModuleList(toPoint_list)

    def forward(self, z, phase, alpha):
        batch_size = z.shape[0]
        node = self.fc1(z).view(batch_size, -1, self.base_size)

        tree_node_list = []
        tree_node_list.append(node)

        for upconv in self.upconv_list[:phase]:
            node = upconv(node)
            tree_node_list.append(node)

        if phase == 0:
            pc = self.toPoints[phase](tree_node_list[-1])
        elif phase == self.nlevels:
            pc = self.final_conv(tree_node_list[-1])
        else:
            prev_phase_node = tree_node_list[-2]
            prev_pc = self.toPoints[phase - 1](prev_phase_node)
            curr_phase_node = tree_node_list[-1]
            if phase == self.nlevels - 1:
                curr_pc = self.final_conv(curr_phase_node)
            else:
                curr_pc = self.toPoints[phase](curr_phase_node)

            if alpha < 1.0:
                pc = nn.Upsample(scale_factor=int(self.tarch[phase]), mode='nearest')(prev_pc) * (1 - alpha) + curr_pc * alpha
            else:
                pc = curr_pc

        pc = torch.transpose(pc, 1, 2).contiguous()
        return pc