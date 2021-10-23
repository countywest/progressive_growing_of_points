from utils.args import Arguments
from utils import get_train_manager

if __name__ == "__main__":
    config = Arguments().prepare_config()
    trainer = get_train_manager(config)
    trainer.run()