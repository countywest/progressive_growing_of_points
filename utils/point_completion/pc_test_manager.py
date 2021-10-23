import torch
import csv
import time
import datetime
from utils.test_manager import TestManager
from tqdm import tqdm
from utils.utils import *
from utils.visu_utils import *
from utils.eval_utils import MetricWriter, FScoreWriter
from models.auto_encoder import get_autoencoder
from termcolor import colored

class PCTestManager(TestManager):
    def __init__(self, config):
        # config
        self.config = config
        self.result_dir, self.plots_dir = self.result_init()
        self.best_ckpt_path = self.get_ckpt_path()

        # data
        self.test_dataset = get_dataset(self.config, type='test')
        self.test_loader, self.num_test = get_comp_dataloader(self.config, self.test_dataset, type='test')

        # models
        self.model = get_autoencoder(self.config)
        self.is_pg = self.config['train_setting']['pg_on']
        if self.is_pg:
            self.phase = self.model.decoder.nlevels
            self.alpha = 0.0 # dummy

        self.print_configs()

    def print_configs(self):
        super(PCTestManager, self).print_configs()
        if self.config['model']['decoder']['type'] == 'topnet':
            print(colored('num_nodes in TopNet: ' + str(self.model.decoder.num_nodes), 'grey', 'on_green'))

    def get_ckpt_path(self):
        log_root_path = os.path.join('logs', self.config['model']['type'])
        model_id = self.config['model']['id']
        log_dir = os.path.join(log_root_path, model_id)
        ckpts_dir = os.path.join(log_dir, 'ckpts')

        ckpts_list = os.listdir(ckpts_dir)
        assert len(ckpts_list) >= 2, colored("There is no checkpoint OR best checkpoint is not generated.", 'white',
                                             'on_red')
        # restore best valid loss
        ckpts_list = sorted(ckpts_list)
        best_ckpt_path = os.path.join(ckpts_dir, ckpts_list[-1])
        return best_ckpt_path

    def restore_model(self):
        best_epoch = self.model.load_ae(self.best_ckpt_path)
        return self.model

    def get_id_and_data(self, data):
        # id & data(partial, gt)
        input = data['partial'].cuda()
        id = data['id'][0]
        synset_id, model_id = id.split('_')
        gt = data['gt'].cuda()

        return input, synset_id, model_id, gt

    def run(self):
        test_metrics = ['cd', 'emd', 'f_score']

        # restore & load model on GPU
        self.model = self.restore_model().cuda()

        results_csv = open(os.path.join(self.result_dir, 'results.csv'), 'w')
        csv_writer = csv.writer(results_csv, delimiter=',', quotechar='"')

        csv_header = ['id']
        test_metric_writers = []
        for metric in test_metrics:
            if metric == 'f_score':
                f_score_percent_list = ['1', '2', '3']
                writer = FScoreWriter(csv_header, f_score_percent_list)
            else:
                writer = MetricWriter(csv_header, metric)
            test_metric_writers.append(writer)

        csv_writer.writerow(csv_header)

        test_start_time = time.time()
        self.model.eval()

        print(colored("Testing...", 'white', 'on_magenta'))
        for i, data in tqdm(enumerate(self.test_loader)):
            with torch.no_grad():
                # id & data
                input, synset_id, model_id, gt = self.get_id_and_data(data)
                # inference
                if self.is_pg:
                    recon = self.model(input, self.phase, self.alpha)
                else:
                    recon = self.model(input)

                # report metrics
                temp_csv_row = [synset_id + '_' + model_id]
                for writer in test_metric_writers:
                    writer.run(temp_csv_row, output=recon, gt=gt, synset_id=synset_id)
                csv_writer.writerow(temp_csv_row)

                # visualize
                if i % self.config['test_setting']['visu_period'] == 0:
                    plots_dir = os.path.join(self.plots_dir, synset_id)
                    os.makedirs(plots_dir, exist_ok=True)

                    num_colors = [1, 2, 4, 8, 1] # (input, recon1 recon2 recon3, gt)

                    # input
                    pcds = [input.detach().cpu().numpy().squeeze()]
                    titles = ['input']

                    # recon
                    # for visualizing multi level nodes
                    for i in range(len(num_colors) - 2):
                        if i == 0:
                            pcds.append(recon.detach().cpu().numpy().squeeze())
                        else:
                            pcds.append(pcds[-1])
                        titles.append(str(num_colors[i + 1]))

                    # gt
                    pcds.append(gt.detach().cpu().numpy().squeeze())
                    titles.append('gt')

                    # suptitle
                    suptitle = ""
                    suptitle_list = []
                    for writer in test_metric_writers:
                        writer.write_name_and_latest_loss(suptitle_list)
                    for elem in suptitle_list:
                        suptitle += elem
                        suptitle += " "

                    plot_pcd(self.config, dir=plots_dir, filename='%s.png' % model_id,
                             pcds=pcds, titles=titles, suptitle=suptitle, num_colors=num_colors)

        # report mean to csv
        for synset_id in test_metric_writers[0].loss_per_cat.keys():
            last_csv_row = [str(synset_id)]
            for writer in test_metric_writers:
                writer.write_mean_per_cat(last_csv_row, synset_id)
            csv_writer.writerow(last_csv_row)
        results_csv.close()

        # summarize results
        with open(os.path.join(self.result_dir, 'results_summary.txt'), 'w') as log:
            for writer in test_metric_writers:
                writer.write_mean_to_log(log, len(self.test_loader))

            log.write('Summary for each category \n')
            header = ['id']
            for writer in test_metric_writers:
                writer.write_name(header)

            header_csv = ""
            for elem in header:
                header_csv += elem
                header_csv += " "
            log.write(header_csv + '\n')

            for synset_id in test_metric_writers[0].loss_per_cat.keys():
                temp_csv_row = [str(synset_id)]
                for writer in test_metric_writers:
                    writer.write_mean_per_cat(temp_csv_row, synset_id)

                temp_log = ""
                for elem in temp_csv_row:
                    temp_log += elem
                    temp_log += " "
                temp_log += '\n'

                log.write(temp_log)


        # print results
        for writer in test_metric_writers:
            writer.print_mean(len(self.test_loader))


        for writer in test_metric_writers:
            writer.print_mean_all_cat()


        print(colored("Testing Done!", 'white', 'on_magenta'))
        print(colored('Total time: ' + str(datetime.timedelta(seconds=time.time() - test_start_time)), 'white', 'on_magenta'))