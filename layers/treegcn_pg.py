''' 
Code from
https://github.com/seowok/TreeGAN/blob/master/layers/gcn.py
https://github.com/seowok/TreeGAN/blob/master/model/gan_network.py

revised by Hyeontae Son
'''
import torch
import torch.nn as nn
from layers.treegcn import TreeGCNGenerator

class TreeGCNGeneratorPG(TreeGCNGenerator):
    def __init__(self, features, degrees, support):
        super(TreeGCNGeneratorPG, self).__init__(features, degrees, support)
        self.nlevels = self.layer_num
        self.degrees = degrees
        self.num_points_of_phase = self.get_num_points_of_phase()

        toPoint_list = []
        for intermediate_level in range(1, self.nlevels):
            toPoint_list.append(nn.Linear(features[intermediate_level], 3))

        self.toPoints = torch.nn.ModuleList(toPoint_list)

    def get_num_points_of_phase(self):
        # number of node for each level
        num_nodes = []
        for i, up_ratio in enumerate(self.degrees):
            if i == 0:
                num_nodes.append(up_ratio)
            else:
                last_num_node = num_nodes[-1]
                num_nodes.append(up_ratio * last_num_node)
        return num_nodes


    def forward(self, z, phase, alpha):
        if len(z.shape) == 2: # (batch_size, noise_dim)
            z = z.unsqueeze(1) # (batch_size, 1, noise_dim)

        batch_size = z.shape[0]
        tree = [z]
        for treegcn in self.gcn[:phase+1]:
            tree = treegcn(tree)

        if phase == 0:
            pc = self.toPoints[phase](tree[-1])
        elif phase == self.nlevels:
            pc = tree[-1]
        else:
            prev_phase_node = tree[-2]
            prev_pc = self.toPoints[phase - 1](prev_phase_node)
            curr_phase_node = tree[-1]
            if phase == self.nlevels - 1:
                curr_pc = tree[-1]
            else:
                curr_pc = self.toPoints[phase](curr_phase_node)

            if alpha < 1.0:
                nn_upsampled_prev_pc = prev_pc.repeat(1, 1, int(self.degrees[phase])).view(batch_size, -1, 3).contiguous()
                pc = nn_upsampled_prev_pc * (1 - alpha) + curr_pc * alpha
            else:
                pc = curr_pc

        self.pointcloud = pc

        return pc
