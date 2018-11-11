import argparse
from validation_loss_analysis import analyse_validation_loss
import configparser
import os


def get_user_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", help="Configuration path.")
    parser.add_argument("-t", "--tasks", help="Tasks in results", default=1)

    return parser.parse_args()


if __name__ == "__main__":

    args = get_user_params()

    config = configparser.RawConfigParser()
    config.read(args.config_path)

    model_dir = config.get('SYSTEM', 'model_dir')
    log_path = model_dir

    ckpoint_path = os.path.join(model_dir, 'models')
    training_iter = analyse_validation_loss(log_path, ckpoint_path, int(args.tasks))
    print('Inference on {} iteration'.format(training_iter))