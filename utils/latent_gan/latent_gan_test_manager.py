import torch
import numpy as np
import time
import datetime
from utils.test_manager import TestManager
from utils.utils import *
from utils.eval_utils import jsd_between_point_cloud_sets, coverage, minimum_mathing_distance, nearest_neighbor_accuracy
from utils.visu_utils import *
from models.latent_gan import get_LatentGAN
from termcolor import colored

class LatentGANTestManager(TestManager):
    def __init__(self, config):
        # config
        self.config = config
        self.result_dir, self.plots_dir = self.result_init()
        self.best_ckpt_path = self.get_best_ckpt_path()
        self.num_evals = self.config['test_setting']['num_evals']

        # data
        self.test_dataset = get_dataset(self.config, type='test')
        self.test_loader = get_dataloader(self.config, self.test_dataset, type='test')
        self.batch_size = self.config['test_setting']['batch_size']

        # models
        self.noise_dim = self.config['model']['noise_dim']
        self.model = get_LatentGAN(self.config)
        self.ckpts_dir_of_AE = os.path.join('logs', 'auto_encoder', self.config['model']['AE_id'], 'ckpts')

        # metrics
        self.JSD = jsd_between_point_cloud_sets
        self.COV = coverage
        self.MMD = minimum_mathing_distance
        self.NNA = nearest_neighbor_accuracy

        self.print_configs()

    def print_configs(self):
        super(LatentGANTestManager, self).print_configs()
        if self.config['model']['decoder']['type'] == 'topnet':
            print(colored('num_nodes in TopNet: ' + str(self.model.decoder.num_nodes), 'grey', 'on_green'))

    def get_best_ckpt_path(self):
        log_root_path = os.path.join('logs', self.config['model']['type'])
        model_id = self.config['model']['id']
        log_dir = os.path.join(log_root_path, model_id)
        ckpts_dir = os.path.join(log_dir, 'ckpts')

        # return best checkpoint path
        ckpts_list = os.listdir(ckpts_dir)
        assert len(ckpts_list) >= 2, colored("There is no checkpoint OR best checkpoint is not generated.", 'white', 'on_red')
        ckpts_list = sorted(ckpts_list)
        best_ckpt_path = os.path.join(ckpts_dir, ckpts_list[-1]) # best.pth
        return best_ckpt_path

    def restore_decoder(self):
        print(colored("Restore pretrained parameters(decoder)...", 'grey', 'on_green'))
        ae_ckpts_list = os.listdir(self.ckpts_dir_of_AE)
        assert len(ae_ckpts_list) >= 2, colored("There is no checkpoint OR best checkpoint is not generated.", 'white', 'on_red')
        ae_ckpts_list = sorted(ae_ckpts_list)
        best_step = self.model.load_decoder(os.path.join(self.ckpts_dir_of_AE, ae_ckpts_list[-1]))
        print(colored("step " + str(best_step) + " chosen... ", 'grey', 'on_green'))
        return best_step

    def restore_generator(self):
        print(colored("Restore pretrained parameters(generator)...", 'grey', 'on_green'))
        best_epoch = self.model.load_generator(self.best_ckpt_path)
        print(colored("epoch " + str(best_epoch) + " chosen... ", 'grey', 'on_green'))
        return best_epoch

    def get_pointclouds(self, decoder, gfvs):
        if self.config['model']['decoder']['is_trained_progressively']:
            last_phase = self.model.decoder.nlevels
            sample_pcs = decoder(gfvs, last_phase, 0)
        else:
            sample_pcs = decoder(gfvs)
        return sample_pcs

    def run(self):
        def iterate_in_chunks(l, n):
            '''Yield successive 'n'-sized chunks from iterable 'l'.
            Note: last chunk will be smaller than l if n doesn't divide l perfectly.
            '''
            for i in range(0, len(l), n):
                yield l[i:i + n]

        # load pretrained decoder & generator(l-GAN)
        self.restore_decoder()
        self.restore_generator()

        # load models on GPU
        decoder = self.model.decoder = self.model.decoder.cuda()
        generator = self.model.generator = self.model.generator.cuda()

        decoder.eval()
        generator.eval()

        ref_pcs = []
        for i, data in enumerate(self.test_loader):
            ref_pcs.append(data['pos'].cuda())
        ref_pcs = torch.cat(ref_pcs)

        test_start_time = time.time()
        print(colored("Testing...", 'white', 'on_magenta'))

        jsd_list=[]
        cov_cd_list=[]
        cov_emd_list=[]
        mmd_cd_list=[]
        mmd_emd_list=[]
        nna_cd_list=[]
        nna_emd_list=[]
        for i in range(self.num_evals):
            with torch.no_grad():
                num_pcs_in_test_set = len(self.test_loader)
                z = torch.randn(num_pcs_in_test_set, self.noise_dim).cuda()
                sample_pcs_list=[]
                for z_sample_chunk in iterate_in_chunks(z, self.batch_size):
                    sample_chunk_gfvs = generator(z_sample_chunk)
                    sample_chunk_pcs = self.get_pointclouds(decoder, sample_chunk_gfvs)
                    sample_pcs_list.append(sample_chunk_pcs)
                sample_pcs = torch.cat(sample_pcs_list)

            
            # jsd
            print(colored("calculating jsd...", 'white', 'on_magenta'))
            jsd = self.JSD(sample_pcs, ref_pcs)
            jsd_list.append(jsd)
            print(colored("JSD: %.8f" % jsd, 'grey', 'on_green'))

            
            # coverage
            print(colored("calculating cov...", 'white', 'on_magenta'))
            cov_cd, _ = self.COV(sample_pcs, ref_pcs, self.batch_size)
            cov_emd, _ = self.COV(sample_pcs, ref_pcs, self.batch_size, use_EMD=True)
            cov_cd_list.append(cov_cd)
            cov_emd_list.append(cov_emd)
            print(colored("COV_CD: %.8f , COV_EMD: %.8f" % (cov_cd, cov_emd), 'grey', 'on_green'))

            # mmd
            print(colored("calculating mmd...", 'white', 'on_magenta'))
            mmd_cd, _ = self.MMD(sample_pcs, ref_pcs, self.batch_size)
            mmd_emd, _ = self.MMD(sample_pcs, ref_pcs, self.batch_size, use_EMD=True)
            mmd_cd_list.append(mmd_cd)
            mmd_emd_list.append(mmd_emd)
            print(colored("MMD_CD: %.8f , MMD_EMD: %.8f" % (mmd_cd, mmd_emd), 'grey', 'on_green'))
            
            # 1-NNA
            print(colored("calculating 1-NNA...", 'white', 'on_magenta'))
            nna_cd = self.NNA(sample_pcs, ref_pcs, self.batch_size)
            nna_emd = self.NNA(sample_pcs, ref_pcs, self.batch_size, use_EMD=True)
            nna_cd_list.append(nna_cd)
            nna_emd_list.append(nna_emd)
            print(colored("1-NNA_CD: %.8f , 1-NNA_EMD: %.8f" % (nna_cd, nna_emd), 'grey', 'on_green'))
            print()

        # report to file
        with open(os.path.join(self.result_dir, 'results_summary.txt'), 'w') as log:
            # average mean
            log.write('average eval\n')
            log.write('JSD: %.8f \n' % np.mean(jsd_list))
            log.write('COV_CD: %.8f , COV_EMD: %.8f \n' % (np.mean(cov_cd_list), np.mean(cov_emd_list)))
            log.write('MMD_CD: %.8f , MMD_EMD: %.8f \n' % (np.mean(mmd_cd_list), np.mean(mmd_emd_list)))
            log.write('1-NNA_CD: %.8f , 1-NNA_EMD: %.8f \n\n' % (np.mean(nna_cd_list), np.mean(nna_emd_list)))

            # each case
            for i in range(len(jsd_list)):
                log.write('JSD: %.8f \n' % jsd_list[i])
                log.write('COV_CD: %.8f , COV_EMD: %.8f \n' % (cov_cd_list[i], cov_emd_list[i]))
                log.write('MMD_CD: %.8f , MMD_EMD: %.8f \n' % (mmd_cd_list[i], mmd_emd_list[i]))
                log.write('1-NNA_CD: %.8f , 1-NNA_EMD: %.8f \n' % (nna_cd_list[i], nna_emd_list[i]))
            

        print(colored("JSD: %.8f" % np.mean(jsd_list), 'grey', 'on_yellow'))
        print(colored("COV_CD: %.8f , COV_EMD: %.8f" % (np.mean(cov_cd_list), np.mean(cov_emd_list)), 'grey', 'on_yellow'))
        print(colored("MMD_CD: %.8f , MMD_EMD: %.8f" % (np.mean(mmd_cd_list), np.mean(mmd_emd_list)), 'grey', 'on_yellow'))
        print(colored("1-NNA_CD: %.8f , 1-NNA_EMD: %.8f" % (np.mean(nna_cd_list), np.mean(nna_emd_list)), 'grey', 'on_yellow'))

        print(colored("Testing Done!", 'white', 'on_magenta'))
        print(colored('Total time: ' + str(datetime.timedelta(seconds=time.time() - test_start_time)), 'white',
                      'on_magenta'))