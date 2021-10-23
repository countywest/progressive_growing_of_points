import torch
import os
import shutil
import glob
import yaml
from layers.pointnet import pointnet
from layers.fc import fc
from layers.srtdecoder import SRTDecoder
from layers.srtdecoder_pg import SRTDecoderPG
from layers.mrtdecoder import MRTDecoder
from layers.mrtdecoder_pg import MRTDecoderPG
from layers.mrtdecoder_pg2 import MRTDecoderPGV2
from layers.topnet import topnet
from layers.topnet_pg import topnetPG
from layers.treegcn import TreeGCNGenerator
from layers.treegcn_pg import TreeGCNGeneratorPG
from torch.autograd import grad

class LatentGAN:
    def __init__(self, config):
        # l-GAN
        self.generator = self.get_generator(config)
        self.discriminator = self.get_discriminator(config)

        # pretrained-AE
        self.encoder = self.get_encoder(config)
        self.decoder = self.get_decoder(config)

    def get_generator(self, config):
        if config['model']['generator']['type'] == 'fc':
            noise_dim = config['model']['noise_dim']
            first_layer_dim = config['model']['generator']['fc_hp']['dims'][0]
            assert noise_dim == first_layer_dim, 'noise dim MUST be equal to the first layer dimension.'
            kwargs = {'dims': config['model']['generator']['fc_hp']['dims']}
            return fc(**kwargs)
        else:
            raise NotImplementedError

    def get_discriminator(self, config):
        if config['model']['discriminator']['type'] == 'fc':
            kwargs = {'dims': config['model']['discriminator']['fc_hp']['dims']}
            return fc(**kwargs)
        else:
            raise NotImplementedError

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

    def get_aeconfig(self, config):
        ae_id = config['model']['AE_id']
        ae_log_directory = os.path.join('logs', 'auto_encoder', ae_id)
        yaml_file_list = glob.glob(os.path.join(ae_log_directory, '*.yaml'))
        assert len(yaml_file_list) == 1, "config file(.yaml) should be ONE in log directory."
        ae_config_path = yaml_file_list[0]
        ae_config = yaml.load(open(ae_config_path), Loader=yaml.FullLoader)
        return ae_config

    def get_decoder(self, config):
        ae_config = self.get_aeconfig(config)
        config['model']['decoder']['is_trained_progressively'] = ae_config['train_setting']['pg_on']

        if config['model']['decoder']['type'] == 'srtdecoder':
            kwargs = {
                'z_dim': config['model']['encoder']['GFV_dim'],
                'nlevels': config['model']['decoder']['srtdecoder_hp']['nlevels'],
                'feat_dims': config['model']['decoder']['srtdecoder_hp']['feat_dims'],
                'num_output_points': config['model']['decoder']['srtdecoder_hp']['num_output_points']
            }
            if config['model']['decoder']['is_trained_progressively']:
                return SRTDecoderPG(**kwargs)
            else:
                return SRTDecoder(**kwargs)
        elif config['model']['decoder']['type'] == 'mrtdecoder':
            kwargs = {
                'z_dim': config['model']['encoder']['GFV_dim'],
                'nlevels': config['model']['decoder']['mrtdecoder_hp']['nlevels'],
                'feat_dims': config['model']['decoder']['mrtdecoder_hp']['feat_dims'],
                'num_output_points': config['model']['decoder']['mrtdecoder_hp']['num_output_points']
            }
            if config['model']['decoder']['is_trained_progressively']:
                return MRTDecoderPG(**kwargs)
            else:
                return MRTDecoder(**kwargs)
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
            if config['model']['decoder']['is_trained_progressively']:
                return topnetPG(**kwargs)
            else:
                return topnet(**kwargs)
        elif config['model']['decoder']['type'] == 'treegcn':
            kwargs = {
                'features': config['model']['decoder']['treegcn_hp']['G_FEAT'],
                'degrees': config['model']['decoder']['treegcn_hp']['DEGREE'],
                'support': config['model']['decoder']['treegcn_hp']['support']
            }
            if config['model']['decoder']['is_trained_progressively']:
                return TreeGCNGeneratorPG(**kwargs)
            else:
                return TreeGCNGenerator(**kwargs)
        else:
            raise NotImplementedError

    def load_encoder(self, ckpt_path):
        data = torch.load(ckpt_path)
        self.encoder.load_state_dict(data['encoder_weights'])
        global_step = data['global_step']
        return global_step

    def load_decoder(self, ckpt_path):
        data = torch.load(ckpt_path)
        self.decoder.load_state_dict(data['decoder_weights'])
        global_step = data['global_step']
        return global_step

    def load_generator(self, ckpt_path):
        data = torch.load(ckpt_path)
        self.generator.load_state_dict(data['generator_weights'])
        epoch = data['epoch']
        return epoch

    def load_discriminator(self, ckpt_path):
        data = torch.load(ckpt_path)
        self.discriminator.load_state_dict(data['discriminator_weights'])
        epoch = data['epoch']
        return epoch

    def load_gan(self, ckpt_path):
        data = torch.load(ckpt_path)
        self.generator.load_state_dict(data['generator_weights'])
        self.discriminator.load_state_dict(data['discriminator_weights'])
        epoch = data['epoch']
        return epoch

    def save_gan(self, epoch, best_valid_loss, is_best, ckpt_dir):
        saving_contents = {
            'epoch': epoch,
            'generator_weights': self.generator.state_dict(),
            'discriminator_weights': self.discriminator.state_dict(),
            'best_valid_loss': best_valid_loss
        }
        torch.save(saving_contents, os.path.join(ckpt_dir, '%04d' % epoch + '.pth'))
        if is_best:
            src_path = os.path.join(ckpt_dir, '%04d' % epoch + '.pth')
            target_path = os.path.join(ckpt_dir, 'best.pth')
            shutil.copy(src_path, target_path)

def get_LatentGAN(config):
    return LatentGAN(config)

class GradientPenalty:
    def __init__(self, lambdaGP):
        self.lambdaGP = lambdaGP

    def __call__(self, netD, real_data, fake_data):
        batch_size = real_data.size(0)

        fake_data = fake_data[:batch_size]

        alpha = torch.rand(batch_size, 1, requires_grad=True).cuda()
        # randomly mix real and fake data
        interpolates = real_data + alpha * (fake_data - real_data)
        # compute output of D for interpolated input
        disc_interpolates = netD(interpolates)
        # compute gradients w.r.t the interpolated outputs

        gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                         grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                         create_graph=True, retain_graph=True, only_inputs=True)[0].contiguous().view(batch_size, -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambdaGP

        return gradient_penalty
