from utils.args import TestArguments
from utils import get_test_manager

if __name__ == "__main__":
    config = TestArguments().prepare_config()
    tester = get_test_manager(config)
    tester.run()