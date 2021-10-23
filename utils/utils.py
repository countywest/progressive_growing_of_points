import torch
import numpy as np
import os
import sys
import yaml
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'ChamferDistancePytorch/chamfer3D'))
sys.path.append(os.path.join(BASE_DIR, 'emd/'))
from datasets import SingleLODShapeNet # auto-encoding
from datasets import lmdb_dataflow, TopNetDataset, PCNTestDataset # point-completion
from dist_chamfer_3D import chamfer_3DFunction
from emd_module import emdModule
earth_mover_module = emdModule()

def chamfer(xyz1, xyz2):
    xyz1 = xyz1.contiguous()
    xyz2 = xyz2.contiguous()
    dist1_square, dist2_square, idx1, idx2 = chamfer_3DFunction.apply(xyz1, xyz2)
    dist1 = torch.mean(torch.sqrt(dist1_square))
    dist2 = torch.mean(torch.sqrt(dist2_square))
    return (dist1 + dist2) / 2


def earth_mover(xyz1, xyz2, is_training=True):
    assert xyz1.shape[1] == xyz2.shape[1]
    if is_training:
        dis, assignment = earth_mover_module(xyz1, xyz2, 0.005, 50)
    else:
        dis, assignment = earth_mover_module(xyz1, xyz2, 0.002, 10000)
    return torch.mean(torch.sqrt(dis))


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
    elif config['dataset']['type'] == 'surreal':
        target_data_path = os.path.join(config['root_data_path'], type)
        return SingleLODSurreal(root_data_path=target_data_path)
    elif config['dataset']['type'] == 'smal':
        if config['dataset']['class'] == 'multi':
            target_data_path = os.path.join(config['root_data_path'], type)
            return SingleLODSMAL(root_data_path=target_data_path, is_multi=True)
        else:
            target_data_path = os.path.join(config['root_data_path'], type, config['dataset']['class'])
            return SingleLODSMAL(root_data_path=target_data_path, is_multi=False)
    elif config['dataset']['type'] == 'pcn':
        if type == 'test':
            return PCNTestDataset(config['root_data_path'], config['test_setting']['num_input_points'])
        else:
            raise NotImplementedError
    elif config['dataset']['type'] == 'topnet':
        return TopNetDataset(root_data_path=config['root_data_path'], type=type)
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

def get_comp_dataloader(config, dataset, type='train'):
    # train & valid
    if type == 'train' or type == 'valid':
        batch_size = config['train_setting']['batch_size']

        if config['dataset']['type'] == 'pcn':
            is_training = True if type == 'train' else False
            df, num_df = lmdb_dataflow(
                os.path.join(config['root_data_path'], type + '.lmdb'),
                batch_size, config['train_setting']['num_input_points'],
                config['dataset']['num_gt_points'], is_training=is_training)
            data_loader = df.get_data()
            len_data = num_df
        else:
            shuffle = True if type == 'train' else False
            data_loader = torch.utils.data.DataLoader(dataset,
                                                      batch_size=batch_size,
                                                      num_workers=config['train_setting']['num_workers'],
                                                      shuffle=shuffle,
                                                      pin_memory=True)
            len_data = len(data_loader)
    # test
    else:
        data_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=1,
                                                  num_workers=config['test_setting']['num_workers'],
                                                  shuffle=False,
                                                  pin_memory=True)
        len_data = len(data_loader)
    return data_loader, len_data

def save_pcd(dir, filename, pcd):
    os.makedirs(dir, exist_ok=True)
    np.save(os.path.join(dir, filename), pcd)