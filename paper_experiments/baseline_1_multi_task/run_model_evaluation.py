from paper_experiments import evaluation
import configparser
import os
import inspect
from pathlib import Path

# get cwd
root = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# get inference path
config = configparser.RawConfigParser()
config.read(os.path.join(root, 'config.ini'))
# path where results should have been stored
results_path = config['INFERENCE']['save_seg_dir']


def get_files(task):
    """
    Get inference results for task
    :param task:
    :return:
    """
    return [Path(os.path.join(results_path, f))
            for f in os.listdir(results_path) if task in f]


def perform_experiment_evaluation(task):
    """
    Evaluate results
    :param task:
    :return:
    """
    # get all files
    list_of_files = get_files(task)

    if task is 'regression':
        # calculate mean absolute error across dataset and across binned ages
        total_error, binned_error = evaluation.calculate_mae(list_of_files)


if __name__ == '__main__':
    perform_experiment_evaluation('classification')
    perform_experiment_evaluation('regression')
