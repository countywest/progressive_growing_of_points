import torch.utils.data as data
import os
import torch
import numpy as np
from plyfile import PlyData

def load_ply(path):
    ply_data = PlyData.read(path)
    vertices = ply_data['vertex']
    points = np.vstack([vertices['x'], vertices['y'], vertices['z']]).astype(np.float32).T
    return points

def load_pth(path):
    pth_data = torch.load(path)
    id = pth_data['id']
    points = pth_data['data']
    return id, points

class SingleLODShapeNet(data.Dataset):
    def __init__(self, root_data_path, is_multi):
        self.data_path_list = self.make_list(root_data_path, is_multi)

    def __len__(self):
        return len(self.data_path_list)

    def __getitem__(self, index):
        filename = self.data_path_list[index]
        parsed_list = filename.split('/')
        synset_id = parsed_list[-2]
        obj_id = parsed_list[-1]
        return {'id':synset_id + '_' + obj_id, 'pos':load_ply(filename)}

    def make_list(self, root_data_path, is_multi):
        data_path_list = []
        if is_multi:
            category_list = sorted(os.listdir(root_data_path))
            for category in category_list:
                file_path_list = sorted(os.listdir(os.path.join(root_data_path, category)))
                for filename in file_path_list:
                    data_path_list.append(os.path.join(root_data_path, category, filename))
        else:
            file_path_list = sorted(os.listdir(root_data_path))
            for filename in file_path_list:
                data_path_list.append(os.path.join(root_data_path, filename))

        return data_path_list
