import torch
import numpy as np

from utils.utils import chamfer, earth_mover, f_score
from termcolor import colored

# for logging during test
class MetricWriter():
    def __init__(self, csv_header, metric_name):
        self.metric_name = metric_name

        if metric_name == 'cd':
            self.metric = chamfer
        elif metric_name == 'emd':
            self.metric = earth_mover
        else:
            raise NotImplementedError

        csv_header.append(self.metric_name)
        self.total_loss = 0
        self.loss_per_cat = {}
        self.latest_loss = -1

    def run(self, row, output, gt, synset_id):
        loss = self.metric(output, gt).detach().item()
        self.total_loss += loss

        if not self.loss_per_cat.get(synset_id):
            self.loss_per_cat[synset_id] = []
        self.loss_per_cat[synset_id].append(loss)

        row.append(loss)
        self.latest_loss = loss

        return loss

    def write_name(self, header):
        header.append(self.metric_name)

    def write_name_and_latest_loss(self, list):
        list.append(self.metric_name)
        list.append('%.4f' % self.latest_loss)

    def write_mean_per_cat(self, row, synset_id):
        row.append(str(np.mean(self.loss_per_cat[synset_id])))

    def write_mean_to_log(self, log, num_total_test):
        log.write('Average ' + self.metric_name + ': %.8f \n' % (self.total_loss / num_total_test))

    def print_mean(self, num_total_test):
        print(colored('Average ' + self.metric_name + ': %f' % (self.total_loss / num_total_test), 'grey', 'on_yellow'))

    def print_mean_all_cat(self):
        print(colored(self.metric_name + ' per category', 'grey', 'on_yellow'))
        for synset_id in self.loss_per_cat.keys():
            print(colored('%s %f' % (synset_id, np.mean(self.loss_per_cat[synset_id])), 'grey', 'on_yellow'))

class FScoreWriter():
    def __init__(self, csv_header, f_score_percent_list):
        self.metric_name = 'f_score'
        self.f_score_percent_list = f_score_percent_list

        self.total_loss = 0
        self.total_f_score = {}
        self.f_score_per_cat = {}
        self.latest_f_score = {}
        for percent in f_score_percent_list:
            csv_header.append('F@' + percent)
            self.total_f_score[percent] = 0
            self.f_score_per_cat[percent] = {}

    def run(self, row, output, gt, synset_id):
        for percent in self.f_score_percent_list:
            f = f_score(recon=output, gt=gt, threshold_ratio=float(percent) / 100).detach().item()
            self.total_f_score[percent] += f

            if not self.f_score_per_cat[percent].get(synset_id):
                self.f_score_per_cat[percent][synset_id] = []
            self.f_score_per_cat[percent][synset_id].append(f)

            row.append(f)
            self.latest_f_score[percent] = f

    def write_name(self, header):
        for percent in self.f_score_percent_list:
            header.append('F-score@' + percent + '%')

    def write_name_and_latest_loss(self, list):
        for percent in self.f_score_percent_list:
            list.append('F@' + percent)
            list.append('%.4f' % self.latest_f_score[percent])

    def write_mean_per_cat(self, row, synset_id):
        for percent in self.f_score_percent_list:
            row.append(str(np.mean(self.f_score_per_cat[percent][synset_id])))

    def write_mean_to_log(self, log, num_total_test):
        for percent in self.f_score_percent_list:
            log.write('Average F-score@' + percent + '%: ' + '%.8f \n' % (
                    self.total_f_score[percent] / num_total_test))

    def print_mean(self, num_total_test):
        for percent in self.f_score_percent_list:
            print(colored('Average F-score@' + percent + '%: ' + '%.8f' %
                          (self.total_f_score[percent] / num_total_test), 'grey', 'on_yellow'))

    def print_mean_all_cat(self):
        for percent in self.f_score_percent_list:
            print(colored('F-score@' + percent + '% per category', 'grey', 'on_yellow'))
            for synset_id in self.f_score_per_cat[percent].keys():
                print(colored('%s %f' % (synset_id, np.mean(self.f_score_per_cat[percent][synset_id])), 'grey', 'on_yellow'))