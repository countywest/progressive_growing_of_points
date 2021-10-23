import shutil
from utils.utils import *
from abc import ABCMeta, abstractmethod
from termcolor import colored

class TestManager(metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, config):
        # config
        self.config = config
        self.result_dir, self.plots_dir = self.result_init()
        self.print_configs()

    def result_init(self):
        result_root_path = os.path.join('results', self.config['model']['type'])
        model_type = self.config['model']['type']
        model_id = self.config['model']['id']
        result_dir = os.path.join(result_root_path, model_id)
        plots_dir = os.path.join(result_dir, 'plots')
        config_file_path = os.path.join('logs', model_type, model_id, model_type + '.yaml')

        if os.path.exists(result_dir):
            delete_key = input(colored('%s already exists. Delete? [y (or ender)/N]' % result_dir, 'white', 'on_red'))
            if delete_key == 'y' or delete_key == "":
                os.system('rm -rf %s/*' % result_dir)
                shutil.copy(config_file_path, result_dir)
                os.makedirs(plots_dir)
        else:
            os.makedirs(result_dir)
            shutil.copy(config_file_path, result_dir)
            os.makedirs(plots_dir)

        return result_dir, plots_dir

    def print_configs(self):
        def print_dict(e, depth):
            for key in e.keys():
                if (isinstance(e[key], dict)):
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

