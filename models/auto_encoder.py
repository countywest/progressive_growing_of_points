import torch
import torch.nn as nn
import os
import shutil
from layers.pointnet import pointnet
from layers.srtdecoder import SRTDecoder
from layers.srtdecoder_pg import SRTDecoderPG
from layers.mrtdecoder import MRTDecoder
from layers.mrtdecoder_pg import MRTDecoderPG
from layers.mrtdecoder_pg2 import MRTDecoderPGV2
from layers.topnet import topnet
from layers.topnet_pg import topnetPG
from layers.treegcn import TreeGCNGenerator
from layers.treegcn_pg import TreeGCNGeneratorPG

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

    def get_encoder(self, config):
        if config['model']['encoder']['type'] == "pointnet":
            GFV_dim = config['model']['encoder']['GFV_dim']
            last_feat_dim = config['model']['encoder']['pointnet_hp']['feat_dims_list'][-1]
            assert GFV_dim == last_feat_dim, 'GFV_dim MUST be equal to last feature dimension.'

            kwargs = {
                'feat_dims_list': config['model']['encoder']['pointnet_hp']['feat_dims_list']
            }
            return pointnet(**kwargs)
        else:
            raise NotImplementedError

    def get_decoder(self, config):
        pass

    def load_ae(self, ckpt_path):
        data = torch.load(ckpt_path)
        self.encoder.load_state_dict(data['encoder_weights'])
        self.decoder.load_state_dict(data['decoder_weights'])
        global_step = data['global_step']
        return global_step

    def save_ae(self, global_step, best_valid_loss, is_best, ckpt_dir):
        saving_contents = {
            'global_step': global_step,
            'encoder_weights': self.encoder.state_dict(),
            'decoder_weights': self.decoder.state_dict(),
            'best_valid_loss': best_valid_loss
        }
        torch.save(saving_contents, os.path.join(ckpt_dir, '%06d' % global_step + '.pth'))
        if is_best:
            src_path = os.path.join(ckpt_dir, '%06d' % global_step + '.pth')
            target_path = os.path.join(ckpt_dir, 'best.pth')
            shutil.copy(src_path, target_path)


class AENormal(AE):
    def __init__(self, config):
        super(AENormal, self).__init__()
        self.encoder = self.get_encoder(config)
        self.decoder = self.get_decoder(config)
        self.encoder_type = config['model']['encoder']['type']
        self.decoder_type = config['model']['decoder']['type']

    def get_decoder(self, config):
        if config['model']['decoder']['type'] == 'srtdecoder':
            kwargs = {
                'z_dim': config['model']['encoder']['GFV_dim'],
                'nlevels': config['model']['decoder']['srtdecoder_hp']['nlevels'],
                'feat_dims': config['model']['decoder']['srtdecoder_hp']['feat_dims'],
                'num_output_points': config['model']['decoder']['srtdecoder_hp']['num_output_points']
            }
            return SRTDecoder(**kwargs)
        elif config['model']['decoder']['type'] == 'mrtdecoder':
            kwargs = {
                'z_dim': config['model']['encoder']['GFV_dim'],
                'nlevels': config['model']['decoder']['mrtdecoder_hp']['nlevels'],
                'feat_dims': config['model']['decoder']['mrtdecoder_hp']['feat_dims'],
                'num_output_points': config['model']['decoder']['mrtdecoder_hp']['num_output_points']
            }
            return MRTDecoder(**kwargs)
        elif config['model']['decoder']['type'] == 'topnet':
            kwargs = {
                'z_dim': config['model']['encoder']['GFV_dim'],
                'nlevels': config['model']['decoder']['topnet_hp']['nlevels'],
                'node_feat_dim': config['model']['decoder']['topnet_hp']['node_feat_dim'],
                'num_output_points': config['model']['decoder']['topnet_hp']['num_output_points']
            }
            return topnet(**kwargs)
        elif config['model']['decoder']['type'] == 'treegcn':
            kwargs = {
                'features': config['model']['decoder']['treegcn_hp']['G_FEAT'],
                'degrees': config['model']['decoder']['treegcn_hp']['DEGREE'],
                'support': config['model']['decoder']['treegcn_hp']['support']
            }
            return TreeGCNGenerator(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, x):  # x: (B, N, 3)
        batch_size, num_points = x.shape[0], x.shape[1]
        if self.encoder_type == 'pointnet':
            # encoding x
            gfv = self.encoder(x)
        else:
            raise NotImplementedError

        out = self.decoder(gfv)
        return out

class AEPG(AE):
    def __init__(self, config):
        super(AEPG, self).__init__()
        self.encoder = self.get_encoder(config)
        self.decoder = self.get_decoder(config)
        self.encoder_type = config['model']['encoder']['type']
        self.decoder_type = config['model']['decoder']['type']

    def get_decoder(self, config):
        if config['model']['decoder']['type'] == 'srtdecoder':
            kwargs = {
                'z_dim': config['model']['encoder']['GFV_dim'],
                'nlevels': config['model']['decoder']['srtdecoder_hp']['nlevels'],
                'feat_dims': config['model']['decoder']['srtdecoder_hp']['feat_dims'],
                'num_output_points': config['model']['decoder']['srtdecoder_hp']['num_output_points']
            }
            return SRTDecoderPG(**kwargs)
        elif config['model']['decoder']['type'] == 'mrtdecoder':
            kwargs = {
                'z_dim': config['model']['encoder']['GFV_dim'],
                'nlevels': config['model']['decoder']['mrtdecoder_hp']['nlevels'],
                'feat_dims': config['model']['decoder']['mrtdecoder_hp']['feat_dims'],
                'num_output_points': config['model']['decoder']['mrtdecoder_hp']['num_output_points']
            }
            return MRTDecoderPG(**kwargs)
        elif config['model']['decoder']['type'] == 'mrtdecoder_pgv2':
            kwargs = {
                'z_dim': config['model']['encoder']['GFV_dim'],
                'nlevels': config['model']['decoder']['mrtdecoder_hp']['nlevels'],
                'feat_dims': config['model']['decoder']['mrtdecoder_hp']['feat_dims'],
                'num_output_points': config['model']['decoder']['mrtdecoder_hp']['num_output_points']
            }
            return MRTDecoderPGV2(**kwargs)
        elif config['model']['decoder']['type'] == 'topnet':
            kwargs = {
                'z_dim': config['model']['encoder']['GFV_dim'],
                'nlevels': config['model']['decoder']['topnet_hp']['nlevels'],
                'node_feat_dim': config['model']['decoder']['topnet_hp']['node_feat_dim'],
                'num_output_points': config['model']['decoder']['topnet_hp']['num_output_points']
            }
            return topnetPG(**kwargs)
        elif config['model']['decoder']['type'] == 'treegcn':
            kwargs = {
                'features': config['model']['decoder']['treegcn_hp']['G_FEAT'],
                'degrees': config['model']['decoder']['treegcn_hp']['DEGREE'],
                'support': config['model']['decoder']['treegcn_hp']['support']
            }
            return TreeGCNGeneratorPG(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, x, phase, alpha):  # x: (B, N, 3)
        batch_size, num_points = x.shape[0], x.shape[1]
        if self.encoder_type == 'pointnet':
            # encoding x
            gfv = self.encoder(x)
        else:
            raise NotImplementedError

        # decoding
        out = self.decoder(gfv, phase, alpha)

        return out

def get_autoencoder(config):
    if config['train_setting']['pg_on']:
        return AEPG(config)
    else:
        return AENormal(config)