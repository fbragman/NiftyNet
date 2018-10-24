from paper_experiments import evaluation
import configparser
import os
import inspect
import pandas as pd
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


def create_csv_and_save(results, name):
    """
    Csv and save in root
    :param result:
    :param name:
    :return:
    """
    df = pd.DataFrame(data=results)
    fname = name + '.csv'
    fname = os.path.join(root, fname)
    df.to_csv(fname, sep='\t', encoding='utf-8')


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
        results = {'mae': total_error, 'binned_mae': binned_error}
        create_csv_and_save(results, 'age_error')

    elif task is 'classification':
        # calculate classification stats: accuracy, precision and recall
        total_stats, binned_stats, bins = evaluation.calculate_classification(list_of_files)
        create_csv_and_save(total_stats, 'classification_stats')
        create_csv_and_save(binned_stats, 'classification_age_binned_stats')
        create_csv_and_save(bins, 'bins')


if __name__ == '__main__':
    perform_experiment_evaluation('classification')
    perform_experiment_evaluation('regression')
