import torch
import time
import datetime
from utils.train_manager import TrainManager
from tqdm import tqdm
from utils.utils import *
from utils.visu_utils import *
from utils.eval_utils import jsd_between_point_cloud_sets
from models.latent_gan import get_LatentGAN, GradientPenalty
from termcolor import colored

class LatentGANTrainManager(TrainManager):
    def __init__(self, config):
        # config
        self.config = config
        self.log_dir, self.ckpts_dir, self.plots_dir = self.log_init()
        self.writer = self.tensorboard_init()

        # data
        self.train_dataset, self.valid_dataset = self.get_datasets()
        self.train_loader, self.valid_loader = self.get_dataloaders()

        # model
        self.noise_dim = self.config['model']['noise_dim']
        self.model = get_LatentGAN(self.config)
        self.ckpts_dir_of_AE = os.path.join('logs', 'auto_encoder', self.config['model']['AE_id'], 'ckpts')

        # optimizer
        self.optimizer_G, self.optimizer_D = self.get_optimizers()
        self.loss_type = self.config['train_setting']['loss']['type']
        if self.loss_type == 'w-gan-gp':
            self.GP = GradientPenalty(self.config['train_setting']['loss']['w-gan-gp_hp']['lambdaGP'])

        # metrics
        self.JSD = jsd_between_point_cloud_sets

        self.print_configs()

    def print_configs(self):
        super(LatentGANTrainManager, self).print_configs()
        if self.config['model']['decoder']['is_trained_progressively']:
            decoder_type = self.config['model']['decoder']['type']
            print(colored('num_points(pg) for ' + decoder_type + ' decoder: ' + str(self.model.decoder.num_points_of_phase), 'grey', 'on_green'))

    def get_optimizers(self):
        # for generator
        optim_config = self.config['train_setting']['generator_optimizer']
        lr = optim_config['init_lr']
        beta1 = optim_config['beta1']
        beta2 = optim_config['beta2']
        if optim_config['type'] == 'Adam':
            optimizer_G = torch.optim.Adam(self.model.generator.parameters(), lr=lr, betas=(beta1, beta2))
        else:
            raise NotImplementedError

        # for discriminator
        optim_config = self.config['train_setting']['discriminator_optimizer']
        lr = optim_config['init_lr']
        beta1 = optim_config['beta1']
        beta2 = optim_config['beta2']
        if optim_config['type'] == 'Adam':
            optimizer_D = torch.optim.Adam(self.model.discriminator.parameters(), lr=lr, betas=(beta1, beta2))
        else:
            raise NotImplementedError

        return optimizer_G, optimizer_D

    def restore(self):
        print(colored("Restore pretrained parameters(G, D)...", 'white', 'on_blue'))
        ckpts_list = os.listdir(self.ckpts_dir)
        ckpts_list = sorted(ckpts_list)

        # restore best valid loss
        last_best_data = torch.load(os.path.join(self.ckpts_dir, ckpts_list[-1]))
        best_valid_loss = last_best_data['best_valid_loss']

        # restore latest model & epoch
        latest_epoch = self.model.load_gan(os.path.join(self.ckpts_dir, ckpts_list[-2]))
        print(colored("restart from epoch " + str(latest_epoch) + "... ", 'white', 'on_blue'))
        return best_valid_loss, latest_epoch

    def restore_best_encoder(self):
        print(colored("Restore pretrained parameters(encoder)...", 'grey', 'on_green'))
        ae_ckpts_list = os.listdir(self.ckpts_dir_of_AE)
        assert len(ae_ckpts_list) >= 2, colored("There is no checkpoint OR best checkpoint is not generated.", 'white',
                                             'on_red')
        ae_ckpts_list = sorted(ae_ckpts_list)
        best_step = self.model.load_encoder(os.path.join(self.ckpts_dir_of_AE, ae_ckpts_list[-1]))
        print(colored("step " + str(best_step) + " chosen... ", 'grey', 'on_green'))
        return best_step

    def restore_best_decoder(self):
        ae_ckpts_list = os.listdir(self.ckpts_dir_of_AE)
        assert len(ae_ckpts_list) >= 2, colored("There is no checkpoint OR best checkpoint is not generated.", 'white',
                                             'on_red')
        ae_ckpts_list = sorted(ae_ckpts_list)
        best_step = self.model.load_decoder(os.path.join(self.ckpts_dir_of_AE, ae_ckpts_list[-1]))
        return best_step

    def get_d_loss(self, d_real_logits, d_fake_logits):
        if self.loss_type == 'vanilla-gan':
            d_real_prob = torch.sigmoid(d_real_logits)
            d_fake_prob = torch.sigmoid(d_fake_logits)
            d_loss = torch.mean(-torch.log(d_real_prob) - torch.log(1.0 - d_fake_prob))
        elif self.loss_type == 'ls-gan':
            d_loss = torch.mean(((d_real_logits - 1.0) ** 2 + d_fake_logits ** 2) / 2.0)
        elif self.loss_type == 'w-gan-gp':
            d_loss = - torch.mean(d_real_logits) + torch.mean(d_fake_logits)
        else:
            raise NotImplementedError
        return d_loss
    
    def get_gp_loss(self, discriminator, real_gfv, fake_gfv):
        return self.GP(discriminator, real_gfv, fake_gfv)

    def get_g_loss(self, g_fake_logits):
        if self.loss_type == 'vanilla-gan':
            g_fake_prob = torch.sigmoid(g_fake_logits)
            g_loss = torch.mean(-torch.log(g_fake_prob))
        elif self.loss_type == 'ls-gan':
            g_loss = torch.mean(((g_fake_logits - 1.0) ** 2) / 2.0)
        elif self.loss_type == 'w-gan-gp':
            g_loss = -torch.mean(g_fake_logits)
        else:
            raise NotImplementedError
        return g_loss

    def get_pointclouds(self, decoder, gfvs):
        if self.config['model']['decoder']['is_trained_progressively']:
            last_phase = self.model.decoder.nlevels
            sample_pcs = decoder(gfvs, last_phase, 0)
        else:
            sample_pcs = decoder(gfvs)
        return sample_pcs

    def run(self):
        # load pretrained encoder
        self.restore_best_encoder()

        # load models on GPU
        encoder = self.model.encoder = self.model.encoder.cuda()
        generator = self.model.generator = self.model.generator.cuda()
        discriminator = self.model.discriminator = self.model.discriminator.cuda()

        best_valid_loss = 10000000
        start_epoch = 0
        max_epoch = self.config['train_setting']['max_epoch']
        if self.config['restore']:
            best_valid_loss, start_epoch = self.restore()
            z_for_vis = torch.from_numpy(np.load(os.path.join(self.log_dir, 'z_for_vis.npy')))
        else:
            num_shapes_per_vis = self.config['train_setting']['num_shapes_per_vis']
            z_for_vis = torch.randn(num_shapes_per_vis, self.noise_dim)
            np.save(os.path.join(self.log_dir, 'z_for_vis'), z_for_vis.cpu().numpy())
        train_start_time = time.time()


        print(colored("Training...", 'white', 'on_blue'))
        for epoch in tqdm(range(max_epoch + 1)):
            if epoch < start_epoch:
                continue
            if epoch != 0 and epoch % self.config['train_setting']['epoch_per_eval'] == 0:
                valid_loss = self.eval(epoch, z_for_vis)
                is_best = True if valid_loss < best_valid_loss else False
                best_valid_loss = min(best_valid_loss, valid_loss)
                self.model.save_gan(epoch=epoch,
                                    best_valid_loss=best_valid_loss, is_best=is_best,
                                    ckpt_dir=self.ckpts_dir)
            if epoch == max_epoch:
                break

            encoder.eval()
            generator.train()
            discriminator.train()

            for i, data in enumerate(self.train_loader):
                # get real GFV
                input = data['pos'].cuda()
                batch_size, num_points = input.shape[0], input.shape[1]
                real_gfv = encoder(input)

                # train D
                z = torch.randn(batch_size, self.noise_dim).cuda()
                with torch.no_grad():
                    fake_gfv = generator(z)

                d_real_logits = discriminator(real_gfv)
                d_fake_logits = discriminator(fake_gfv)
                d_loss = self.get_d_loss(d_real_logits, d_fake_logits)

                # gradient penalty
                if self.loss_type == 'w-gan-gp':
                    gp_loss = self.get_gp_loss(discriminator, real_gfv, fake_gfv)
                    d_loss += gp_loss

                self.optimizer_D.zero_grad()
                d_loss.backward()
                self.optimizer_D.step()

                # train G
                z = torch.randn(batch_size, self.noise_dim).cuda()
                fake_gfv = generator(z)
                g_fake_logits = discriminator(fake_gfv)
                g_loss = self.get_g_loss(g_fake_logits)

                self.optimizer_G.zero_grad()
                g_loss.backward()
                self.optimizer_G.step()

                # report to the tensorboard
                loss_name = 'l-GAN_train/' + self.config['train_setting']['loss']['type']
                self.writer.add_scalar(loss_name + '_D_loss', d_loss.item(), epoch * len(self.train_loader) + i)
                self.writer.add_scalar(loss_name + '_G_loss', g_loss.item(), epoch * len(self.train_loader) + i)

        print(colored("Training Done!", 'white', 'on_blue'))
        print(colored('Total time: ' + str(datetime.timedelta(seconds=time.time() - train_start_time)), 'white', 'on_blue'))

    def eval(self, epoch, z_for_vis):
        def iterate_in_chunks(l, n):
            '''Yield successive 'n'-sized chunks from iterable 'l'.
            Note: last chunk will be smaller than l if n doesn't divide l perfectly.
            '''
            for i in range(0, len(l), n):
                yield l[i:i + n]

        self.restore_best_decoder()
        decoder = self.model.decoder = self.model.decoder.cuda() # load decoder on GPU
        generator = self.model.generator

        decoder.eval()
        generator.eval()

        with torch.no_grad():
            num_pcs_in_valid_set = len(self.valid_loader)
            z = torch.randn(num_pcs_in_valid_set, self.noise_dim).cuda()
            sample_pcs_list=[]
            for z_sample_chunk in iterate_in_chunks(z, self.config['train_setting']['batch_size']):
                sample_chunk_gfvs = generator(z_sample_chunk)
                sample_chunk_pcs = self.get_pointclouds(decoder, sample_chunk_gfvs)
                sample_pcs_list.append(sample_chunk_pcs)
            sample_pcs = torch.cat(sample_pcs_list)

        ref_pcs = []
        for i, data in enumerate(self.valid_loader):
            ref_pcs.append(data['pos'].cuda())
        ref_pcs = torch.cat(ref_pcs)

        jsd = self.JSD(sample_pcs, ref_pcs)

        # report to tensorboard
        loss_name = 'l-GAN_valid/' + 'jsd'
        self.writer.add_scalar(loss_name, jsd, epoch)

        # visualization
        z = z_for_vis.cuda()
        sample_gfvs = generator(z)
        sample_pcs = self.get_pointclouds(decoder, sample_gfvs).detach().cpu().numpy()

        pcds = []
        titles = []
        num_colors = [8] * sample_pcs.shape[0]
        for i in range(sample_pcs.shape[0]):
            pcds.append(sample_pcs[i])
            titles.append('')
        plot_pcd(self.config, dir=self.plots_dir, filename='epoch_%06d.png' % epoch,
                 pcds=pcds, titles=titles, num_colors=num_colors)

        return jsd