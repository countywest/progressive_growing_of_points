import torch
import time
import datetime
from utils.train_manager import TrainManager
from tqdm import tqdm
from utils.utils import *
from utils.visu_utils import *
from models.auto_encoder import get_autoencoder
from termcolor import colored
from bisect import bisect

class PCTrainManager(TrainManager):
    def __init__(self, config):
        # config
        self.config = config
        self.log_dir, self.ckpts_dir, self.plots_dir = self.log_init()
        self.writer = self.tensorboard_init()
        self.global_step = 0

        # data
        self.train_dataset, self.valid_dataset = self.get_datasets()
        [self.train_loader, self.num_train], [self.valid_loader, self.num_valid] = self.get_dataloaders()

        # model
        self.model = get_autoencoder(self.config)
        self.is_pg = self.config['train_setting']['pg_on']
        if self.is_pg:
            self.phase = 0
            self.alpha = 0.0
            self.cumulative_steps = self.config['train_setting']['pg_hp']['cumulative_steps']
            self.last_phase = self.model.decoder.nlevels # for eval
            self.num_points_of_phase = self.model.decoder.num_points_of_phase

        # optimizer
        self.optimizer = self.get_optimizer()
        self.loss = chamfer if self.config['train_setting']['loss'] == 'CD' else earth_mover

        self.print_configs()

    def get_datasets(self):
        if self.config['dataset']['type'] == 'pcn':
            return None, None
        elif self.config['dataset']['type'] == 'topnet':
            return get_dataset(self.config, type='train'), get_dataset(self.config, type='valid')
        else:
            raise NotImplementedError

    def get_dataloaders(self):
        return get_comp_dataloader(self.config, self.train_dataset, type='train'), \
               get_comp_dataloader(self.config, self.valid_dataset, type='valid')

    def print_configs(self):
        super(PCTrainManager, self).print_configs()
        # print decoder specification only for progressive growing version.
        if self.is_pg:
            decoder_type = self.config['model']['decoder']['type']
            print(colored('num_points(pg) for ' + decoder_type + ' decoder: ' + str(self.num_points_of_phase), 'grey', 'on_green'))

    def get_optimizer(self):
        optim_config = self.config['train_setting']['optimizer']
        lr = optim_config['init_lr']
        beta1 = optim_config['beta1']
        beta2 = optim_config['beta2']
        if optim_config['type'] == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(beta1, beta2))
        else:
            raise NotImplementedError
        return optimizer

    def restore(self):
        print(colored("Restore pretrained parameters...", 'white', 'on_blue'))
        ckpts_list = os.listdir(self.ckpts_dir)
        assert len(ckpts_list) >= 2, colored("There is no checkpoint OR best checkpoint is not generated.", 'white', 'on_red')

        # restore best valid loss
        ckpts_list = sorted(ckpts_list)
        last_best_data = torch.load(os.path.join(self.ckpts_dir, ckpts_list[-1]))
        best_valid_loss = last_best_data['best_valid_loss']

        # restore latest model & step
        latest_global_step = self.model.load_ae(os.path.join(self.ckpts_dir, ckpts_list[-2]))
        self.global_step = latest_global_step
        print(colored("restart from global_step " + str(latest_global_step) + "... ", 'white', 'on_blue'))

        if self.is_pg:
            # restore phase
            self.phase = bisect(self.cumulative_steps, self.global_step)

            # restore alpha
            if self.phase == 0:
                total_steps_of_current_phase = self.cumulative_steps[0]
                trained_steps_of_current_phase = self.global_step
            else:
                total_steps_of_current_phase = self.cumulative_steps[self.phase] - self.cumulative_steps[self.phase - 1]
                trained_steps_of_current_phase = self.global_step - self.cumulative_steps[self.phase - 1]
            self.alpha = 2.0 * trained_steps_of_current_phase / float(total_steps_of_current_phase)
            print(colored("step: " + str(self.global_step)
                          + ' phase: ' + str(self.phase)
                          + ' alpha: ' + str(self.alpha), 'white', 'on_blue'))
        return best_valid_loss

    def update_phase_and_alpha(self):
        assert self.is_pg, 'update_phase_and_alpha is only for progressive growing version.'
        self.phase = bisect(self.cumulative_steps, self.global_step)
        if self.global_step == self.cumulative_steps[self.phase - 1]:
            self.alpha = 0.0
        else:
            if self.phase == 0:
                total_steps_of_current_phase = self.cumulative_steps[self.phase]
            else:
                total_steps_of_current_phase = self.cumulative_steps[self.phase] - self.cumulative_steps[self.phase - 1]
            self.alpha += 2.0 / total_steps_of_current_phase

    def parse_data(self, data):
        if self.config['dataset']['type'] == 'pcn':
            id, input, gt = data
            input = torch.from_numpy(input)
            gt = torch.from_numpy(gt)
        else:
            id = data['id']
            input = data['partial']
            gt = data['gt']
        return id, input, gt

    def run(self):
        # load model on GPU
        self.model = self.model.cuda()

        best_valid_loss = 10000000
        max_step = self.config['train_setting']['max_step']
        if self.config['restore']:
            best_valid_loss = self.restore()
        train_start_time = time.time()

        print(colored("Training...", 'white', 'on_blue'))
        if self.config['dataset']['type'] == 'pcn':
            train_iter = self.train_loader
        elif self.config['dataset']['type'] == 'topnet':
            train_iter = iter(self.train_loader)
        else:
            raise NotImplementedError


        for step in tqdm(range(self.global_step, max_step + 1)):
            # update global step
            self.global_step = step

            # update phase & alpha for progressive growing version
            if self.is_pg:
                self.update_phase_and_alpha()

            # evaluation & save model
            if step != 0 and step % self.config['train_setting']['steps_per_eval'] == 0:
                if self.is_pg: # calculate valid_loss only when progressive growing ended (to find the best model only for last phase)
                    if step >= (self.cumulative_steps[-1] + self.cumulative_steps[-2]) / 2: 
                        valid_loss = self.eval(step,
                                        self.config['train_setting']['visu_period'],
                                        self.config['train_setting']['visu_period_fine'])
                    else:
                        self.eval(step,
                                        self.config['train_setting']['visu_period'],
                                        self.config['train_setting']['visu_period_fine'])
                        valid_loss = 10000
                else:
                    valid_loss = self.eval(step,
                                       self.config['train_setting']['visu_period'],
                                       self.config['train_setting']['visu_period_fine'])

                is_best = True if valid_loss < best_valid_loss else False
                best_valid_loss = min(best_valid_loss, valid_loss)
                self.model.save_ae(global_step=self.global_step,
                                   best_valid_loss=best_valid_loss, is_best=is_best,
                                   ckpt_dir=self.ckpts_dir)
                print(colored("step: %d, valid_loss(%s): %.8f" % (step, self.config['train_setting']['loss'], valid_loss),
                              'grey', 'on_green'))

            # end of training
            if step == max_step:
                break

            self.model.train()

            # extract data batch
            try:
                data = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                data = next(train_iter)

            id, input, gt = self.parse_data(data)
            # load data to GPU
            input = input.cuda()
            gt = gt.cuda()

            if self.is_pg:
                recon = self.model(input, self.phase, self.alpha)
            else:
                recon = self.model(input)
                
            # loss
            loss = self.loss(recon, gt)

            # backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # report to the tensorboard
            loss_name = 'PC_train/' + self.config['train_setting']['loss']
            self.writer.add_scalar(loss_name, loss.detach().item(), step)

        print(colored("Training Done!", 'white', 'on_blue'))
        print(colored('Total time: ' + str(datetime.timedelta(seconds=time.time() - train_start_time)), 'white', 'on_blue'))

    def eval(self, step, visu_period, visu_period_fine):
        self.model.eval()
        batch_size = self.config['train_setting']['batch_size']
        valid_loss = 0.0
        if self.config['dataset']['type'] == 'pcn':
            self.valid_loader, _ = get_comp_dataloader(self.config, self.valid_dataset, type='valid')
            num_eval_steps = self.num_valid // batch_size
            valid_iter = self.valid_loader
            for i in range(num_eval_steps):
                data = next(valid_iter)
                id, input, gt = self.parse_data(data)
                # load data to GPU
                input = input.cuda()
                gt = gt.cuda()

                if self.is_pg:
                    recon = self.model(input, self.phase, self.alpha)
                else:
                    recon = self.model(input)

                valid_loss += self.loss(recon, gt).detach().item()
                if i % visu_period == 0:
                    recon_num_colors = 8
                    if self.is_pg:
                        recon = self.model(input, self.phase, self.alpha)
                        recon_node_num = self.num_points_of_phase[-1] if self.phase == self.last_phase else \
                        self.num_points_of_phase[self.phase]
                        recon_num_colors = min(recon_node_num, 8)

                    for j in range(0, batch_size, visu_period_fine):
                        model_id = id[j]
                        pcds = [input[j].detach().cpu().numpy().squeeze(),
                                recon[j].detach().cpu().numpy().squeeze(),
                                gt[j].detach().cpu().numpy().squeeze()]
                        titles = ['input', 'recon', 'gt']
                        plot_pcd(self.config, dir=os.path.join(self.plots_dir, model_id), filename='%06d.png' % step,
                                 pcds=pcds, titles=titles, num_colors=[1, recon_num_colors, 1])

            valid_loss /= num_eval_steps

            # report to the tensorboard
            loss_name = 'PC_valid/' + self.config['train_setting']['loss']
            self.writer.add_scalar(loss_name, valid_loss, step)

        elif self.config['dataset']['type'] == 'topnet':
            for i, data in enumerate(self.valid_loader):
                with torch.no_grad():
                    id, input, gt = self.parse_data(data)
                    # load data to GPU
                    input = input.cuda()
                    gt = gt.cuda()

                    if self.is_pg:
                        recon = self.model(input, self.phase, self.alpha)
                    else:
                        recon = self.model(input)

                    valid_loss += self.loss(recon, gt).detach().item()
                if i % visu_period == 0:
                    recon_num_colors = 8
                    if self.is_pg:
                        recon = self.model(input, self.phase, self.alpha)
                        recon_node_num = self.num_points_of_phase[-1] if self.phase == self.last_phase else \
                        self.num_points_of_phase[self.phase]
                        recon_num_colors = min(recon_node_num, 8)

                    for j in range(0, batch_size, visu_period_fine):
                        model_id = id[j].split('.')[0]
                        pcds = [input[j].detach().cpu().numpy().squeeze(),
                                recon[j].detach().cpu().numpy().squeeze(),
                                gt[j].detach().cpu().numpy().squeeze()]
                        titles = ['input', 'recon', 'gt']
                        plot_pcd(self.config, dir=os.path.join(self.plots_dir, model_id), filename='%06d.png' % step,
                                 pcds=pcds, titles=titles, num_colors=[1, recon_num_colors, 1])

            valid_loss /= len(self.valid_loader)

            # report to the tensorboard
            loss_name = 'PC_valid/' + self.config['train_setting']['loss']
            self.writer.add_scalar(loss_name, valid_loss, step)

        else:
            raise NotImplementedError


        return valid_loss
