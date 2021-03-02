import shutil
from utils.utils import *
from tensorboardX import SummaryWriter
from abc import ABCMeta, abstractmethod
from termcolor import colored

class TrainManager(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config):
        self.config = config
        self.log_dir, self.ckpts_dir, self.plots_dir = self.log_init()
        self.writer = self.tensorboard_init()
        self.print_configs()

    def log_init(self):
        log_root_path = os.path.join('logs', self.config['model']['type'])
        model_id = self.config['model']['id']
        log_dir = os.path.join(log_root_path, model_id)
        ckpts_dir = os.path.join(log_dir, 'ckpts')
        plots_dir = os.path.join(log_dir, 'plots')
        config_file_path = os.path.join('configs', self.config['model']['type'] + '.yaml')

        if os.path.exists(log_dir):
            if not self.config['restore']:
                delete_key = input(colored('%s already exists. Delete? [y (or ender)/N]' % log_dir, 'white', 'on_red'))
                if delete_key == 'y' or delete_key == "":
                    os.system('rm -rf %s/*' % log_dir)
                    shutil.copy(config_file_path, log_dir)
                    os.makedirs(ckpts_dir)
                    os.makedirs(plots_dir)
        else:
            os.makedirs(log_dir)
            shutil.copy(config_file_path, log_dir)
            os.makedirs(ckpts_dir)
            os.makedirs(plots_dir)

        return log_dir, ckpts_dir, plots_dir

    def tensorboard_init(self):
        model_id = self.config['model']['id']
        tensorboard_dir = os.path.join(self.log_dir, 'log')
        writer = SummaryWriter(os.path.join(tensorboard_dir))
        return writer

    def print_configs(self):
        def print_dict(e, depth):
            for key in e.keys():
                if(isinstance(e[key], dict)):
                    print("----" * depth + colored(key, 'grey', 'on_green'))
                    print_dict(e[key], depth + 1)
                    print()
                else:
                    print("----" * depth + colored(key + ": " + str(e[key]), 'grey', 'on_green'))

        print(colored("//////////////////////////////////////", 'white', 'on_green'))
        print(colored("///////// EXPERIMENT SETTING /////////", 'green'))
        print(colored("//////////////////////////////////////", 'white', 'on_green'))
        print()
        print_dict(self.config, 0)
        print()

    def get_datasets(self):
        return get_dataset(self.config, type='train'), get_dataset(self.config, type='valid')

    def get_dataloaders(self):
        train_loader = get_dataloader(self.config, self.train_dataset, type='train')
        valid_loader = get_dataloader(self.config, self.valid_dataset, type='valid')
        return train_loader, valid_loader

    def restore(self):
        pass

    def run(self):
        pass