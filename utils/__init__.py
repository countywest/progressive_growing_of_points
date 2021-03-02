from utils.auto_encoder.ae_train_manager import AETrainManager
from utils.auto_encoder.ae_test_manager import AETestManager

def get_train_manager(config):
    model_type = config['model']['type']
    if model_type == "auto_encoder":
        return AETrainManager(config)
    else:
        raise NotImplementedError

def get_test_manager(config):
    model_type = config['model']['type']
    if model_type == "auto_encoder":
        return AETestManager(config)
    else:
        raise NotImplementedError