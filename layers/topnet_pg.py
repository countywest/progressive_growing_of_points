''' 
Code from
https://github.com/lynetcha/completion3d/blob/master/tensorflow/models/TopNet.py

revised by Hyeontae Son
'''
import torch
from torch.nn import Sequential, Tanh, Linear, ModuleList
from layers.topnet import topnet
tree_arch = {}
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]
tree_arch[11] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

class topnetPG(topnet):
    def __init__(self, z_dim, nlevels, node_feat_dim, num_output_points):
        super(topnetPG, self).__init__(z_dim, nlevels, node_feat_dim, num_output_points)
        self.num_points_of_phase = self.num_nodes
        toPoint_list = []
        for intermediate_level in range(self.nlevels - 1):
            toPoint_list.append(Sequential(Linear(self.node_feat_dim, 3), Tanh()))

        self.toPoints = ModuleList(toPoint_list)

    def forward(self, z, phase, alpha):
        batch_size = z.shape[0]
        node = self.fc1(z)
        node = node.view(-1, int(self.tarch[0]), self.node_feat_dim)

        tree_node_list = []
        tree_node_list.append(node)

        for upconv in self.upconv_list[:phase]:
            z_tiled = torch.unsqueeze(z, 1).repeat(1, node.shape[1], 1)
            node = torch.cat((z_tiled, node), dim=2)
            node = upconv(node)
            node = node.view(batch_size, -1, self.node_feat_dim)
            tree_node_list.append(node)

        if phase == 0:
            pc = self.toPoints[phase](tree_node_list[-1])
        elif phase == self.nlevels:
            pc = self.last_mlp(tree_node_list[-1])
        else:
            prev_phase_node = tree_node_list[-2]
            prev_pc = self.toPoints[phase - 1](prev_phase_node)
            curr_phase_node = tree_node_list[-1]
            if phase == self.nlevels - 1:
                curr_pc = self.last_mlp(curr_phase_node)
            else:
                curr_pc = self.toPoints[phase](curr_phase_node)

            if alpha < 1.0:
                nn_upsampled_prev_pc = prev_pc.repeat(1, 1, int(self.tarch[phase])).view(batch_size, -1, 3).contiguous()
                pc = nn_upsampled_prev_pc * (1 - alpha) + curr_pc * alpha
            else:
                pc = curr_pc

        pc = pc.contiguous()
        return pc







