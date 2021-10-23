import argparse
import yaml
import os
import glob
from termcolor import colored

class Arguments():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--category_path', default='configs/category.yaml', help='path of category.yaml')
        self.parser.add_argument('--model_type', default='auto_encoder', help='auto_encoder, point_completion, l-GAN')
        self.parser.add_argument('--model_id', default='test1', help='THIS MUST BE EQUAL TO THE MODEL ID IN config.yaml')
        self.parser.add_argument('--restore', action='store_true')
        self.args = self.parser.parse_args()

    def prepare_config(self):
        def connect_path(config):
            dataset_type = config['dataset']['type']
            num_gt_points = config['dataset']['num_gt_points']
            root_data_path = os.path.join('data', dataset_type + '_' + str(num_gt_points))
            config['root_data_path'] = root_data_path
            return config

        # training from pretrained networks
        if self.args.restore:
            log_directory = os.path.join('logs', self.args.model_type, self.args.model_id)
            yaml_file_list = glob.glob(os.path.join(log_directory, '*.yaml'))
            assert len(yaml_file_list) == 1, "config file(.yaml) should be ONE in log directory."
            config_path = yaml_file_list[0]
        # training from scratch
        else:
            config_path = os.path.join('configs', self.args.model_type + '.yaml')

        config = yaml.load(open(config_path), Loader=yaml.FullLoader)
        assert self.args.model_id == config['model']['id'], \
                    colored('model ids from the argument and the config.yaml are different', 'grey', 'on_green')
        config['restore'] = self.args.restore
        config = connect_path(config)

        return config

class TestArguments(Arguments):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--category_path', default='configs/category.yaml', help='path of category.yaml')
        self.parser.add_argument('--model_type', default='auto_encoder', help='auto_encoder, point_completion, l-GAN')
        self.parser.add_argument('--model_id', default='test1', help='THIS MUST BE EQUAL TO THE MODEL ID IN config.yaml')
        self.parser.add_argument('--sampling', action='store_true')
        self.args = self.parser.parse_args()

    def prepare_config(self):
        def connect_path(config):
            dataset_type = config['dataset']['type']
            num_gt_points = config['dataset']['num_gt_points']
            root_data_path = os.path.join('data', dataset_type + '_' + str(num_gt_points))
            config['root_data_path'] = root_data_path
            return config

        log_directory = os.path.join('logs', self.args.model_type, self.args.model_id)
        yaml_file_list = glob.glob(os.path.join(log_directory, '*.yaml'))
        assert len(yaml_file_list) == 1, "config file(.yaml) should be ONE in log directory."
        config_path = yaml_file_list[0]

        config = yaml.load(open(config_path), Loader=yaml.FullLoader)
        assert self.args.model_id == config['model']['id'], \
                    colored('model ids from the argument and the config.yaml are different', 'grey', 'on_green')
        config['sampling'] = self.args.sampling
        config = connect_path(config)

        return config