from utils.auto_encoder.ae_train_manager import AETrainManager
from utils.auto_encoder.ae_test_manager import AETestManager
from utils.latent_gan.latent_gan_train_manager import LatentGANTrainManager
from utils.latent_gan.latent_gan_test_manager import LatentGANTestManager
from utils.point_completion.pc_train_manager import PCTrainManager
from utils.point_completion.pc_test_manager import PCTestManager

def get_train_manager(config):
    model_type = config['model']['type']
    if model_type == "auto_encoder":
        return AETrainManager(config)
    elif model_type == "l-GAN":
        return LatentGANTrainManager(config)
    elif model_type == "point_completion":
        return PCTrainManager(config)
    else:
        raise NotImplementedError

def get_test_manager(config):
    model_type = config['model']['type']
    if model_type == "auto_encoder":
        return AETestManager(config)
    elif model_type == "l-GAN":
        return LatentGANTestManager(config)
    elif model_type == "point_completion":
        return PCTestManager(config)
    else:
        raise NotImplementedError
