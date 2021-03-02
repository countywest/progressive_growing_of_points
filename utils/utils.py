import torch
import numpy as np
import os
import sys
import yaml
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'ChamferDistancePytorch/chamfer3D'))
sys.path.append(os.path.join(BASE_DIR, 'PyTorchEMD/'))
from datasets import SingleLODShapeNet # auto-encoding
from dist_chamfer_3D import chamfer_3DFunction
from emd import earth_mover_distance

def chamfer(xyz1, xyz2):
    xyz1 = xyz1.contiguous()
    xyz2 = xyz2.contiguous()
    dist1_square, dist2_square, idx1, idx2 = chamfer_3DFunction.apply(xyz1, xyz2)
    dist1 = torch.mean(torch.sqrt(dist1_square))
    dist2 = torch.mean(torch.sqrt(dist2_square))
    return (dist1 + dist2) / 2

def earth_mover(xyz1, xyz2):
    assert xyz1.shape[1] == xyz2.shape[1]
    num_points = torch.tensor(xyz1.shape[1], dtype=torch.float32).cuda()
    cost = earth_mover_distance(xyz1, xyz2, transpose=False)
    return torch.mean(cost / num_points)

def f_score(recon, gt, threshold_ratio=0.01):
    recon = recon.contiguous()
    gt = gt.contiguous()

    x_min = torch.min(gt[:, :, 0])
    x_max = torch.max(gt[:, :, 0])
    y_min = torch.min(gt[:, :, 1])
    y_max = torch.max(gt[:, :, 1])
    z_min = torch.min(gt[:, :, 2])
    z_max = torch.max(gt[:, :, 2])
    avg_side_length = ((x_max - x_min) + (y_max - y_min) + (z_max - z_min)) / 3.0
    threshold = avg_side_length * threshold_ratio

    dist1_square, dist2_square, idx1, idx2 = chamfer_3DFunction.apply(recon, gt)
    dist1 = torch.sqrt(dist1_square)
    dist2 = torch.sqrt(dist2_square)
    precision = torch.mean((dist1 < threshold).float(), dim=1)
    recall = torch.mean((dist2 < threshold).float(), dim=1)
    f_score = 2 * precision * recall / (precision + recall)

    return f_score

def get_dataset(config, type='train'):
    if config['dataset']['type'] == 'shapenet':
        if config['dataset']['class'] == 'multi':
            target_data_path = os.path.join(config['root_data_path'], type)
            return SingleLODShapeNet(root_data_path=target_data_path, is_multi=True)
        else:
            category2synsetid = yaml.load(open(os.path.join('configs', 'category.yaml')), Loader=yaml.FullLoader)
            synsetid = category2synsetid[config['dataset']['class']]
            target_data_path = os.path.join(config['root_data_path'], type, str(synsetid))
            return SingleLODShapeNet(root_data_path=target_data_path, is_multi=False)
    else:
        raise NotImplementedError

def get_dataloader(config, dataset, type='train'):
    if type == 'train' or type == 'valid':
        batch_size = config['train_setting']['batch_size']
        shuffle = True if type == 'train' else False
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=batch_size,
                                                  num_workers=config['train_setting']['num_workers'],
                                                  shuffle=shuffle,
                                                  pin_memory=True)
    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=1,
                                                  num_workers=config['test_setting']['num_workers'],
                                                  shuffle=False,
                                                  pin_memory=True)
    return data_loader

def save_pcd(dir, filename, pcd):
    os.makedirs(dir, exist_ok=True)
    np.save(os.path.join(dir, filename), pcd)

if __name__ == "__main__":
    # CD & EMD test
    p1 = torch.rand(32, 2000, 3).cuda()
    p2 = torch.rand(32, 2000, 3).cuda()

    print("chamfer: " + str(chamfer(p1, p2)))
    print("earth mover: " + str(earth_mover(p1, p2)))