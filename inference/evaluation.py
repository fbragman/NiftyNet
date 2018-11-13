import numpy as np
import nibabel as nib
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from multiprocessing import Process, Manager, Pool

NUM_BINS = 15


def calculate_expectation(results):
    """
    Calculate expectation of results from stochastic passes
    :param results:
    :return:
    """


def calculate_mode(results):
    """
    Calculate mode of results from stochastic passes
    :param results:
    :return:
    """


def calculate_variance(results):
    """
    Calculate variance of results from stochastic passes
    :param results:
    :return:
    """


def calculate_entropy(results, c):
    """
    Calculate entropy of results from p(y=c) results
    :param results: probabilities for argmax(mean(p))
    :param c: argmax(mean(p))
    :return:
    """


def get_ground_truth(files, label):
    """
    Get ground truth since it is in the file name
    :param files:
    :return:
    """
    gt = []
    for f in files:
        image_name = f.name
        age, gender = image_name.split("_")[:2]
        if label is 'age':
            gt.append(age)
        else:
            gt.append(gender)
    gt = np.asarray(gt, dtype=np.float32)
    return gt


def get_data(file):
    """
    Get data via multiprocessing
    :param file:
    :param L:
    :return:
    """
    return np.asscalar(nib.load(file).get_data().squeeze())


def get_prediction_parallel(files):
    """
    Load image and get prediction
    :param files:
    :return:
    """
    pool = Pool(processes=8)
    res = pool.map(get_data, files)
    pool.close()
    pool.join()
    return res



def calculate_classification(files):
    """
    Calculate classification statistics: accuracy, precision, recall
    :param files:
    :return:
    """
    age_gt = get_ground_truth(files, 'age')
    gender_gt = get_ground_truth(files, 'gender')
    pred = get_prediction_parallel(files)

    total_precision = precision_score(gender_gt, pred, 'weighted')
    total_recall = recall_score(gender_gt, pred, 'weighted')
    total_accuracy = accuracy_score(gender_gt, pred, 'weighted')

    total_results = {'precision': [total_precision],
                     'recall': [total_recall],
                     'accuracy': [total_accuracy]
                     }

    # get binned results
    bins = np.linspace(0, np.max(age_gt), NUM_BINS)
    age_bins = np.digitize(age_gt, bins)
    bin_precision = []
    bin_recall = []
    bin_accuracy = []
    for bin in np.unique(age_bins):
        idx = (age_bins == bin)
        bin_precision.append(precision_score(gender_gt[idx], pred[idx], 'weighted'))
        bin_recall.append(recall_score(gender_gt[idx], pred[idx], 'weighted'))
        bin_accuracy.append(accuracy_score(gender_gt[idx], pred[idx], 'weighted'))

    bin_precision = np.asarray(bin_precision)
    bin_recall = np.asarray(bin_recall)
    bin_accuracy = np.asarray(bin_accuracy)

    # get bins
    bin_pairs = dict()
    start_b = []
    end_b = []
    for it, b in enumerate(bins):
        if it < len(bins)-1:
            start_b.append(b)
            end_b.append(bins[it+1])
    bin_pairs['bin_start'] = start_b
    bin_pairs['bin_end'] = end_b

    bin_results = {'precision': bin_precision,
                   'recall': bin_recall,
                   'accuracy': bin_accuracy}

    return total_results, bin_results, bin_pairs


def calculate_mae(files):
    """
    Calculate mean absolute error
    :param files:
    :return:
    """
    gt = get_ground_truth(files, 'age')
    pred = get_prediction_parallel(files)
    # age * 100
    pred = pred * 100
    # calculate errors
    error = np.abs(pred-gt)
    mae = np.mean(error)

    # get binned results
    bins = np.linspace(0, np.max(gt), NUM_BINS)
    age_bins = np.digitize(gt, bins)
    bin_mae = []
    for bin in np.unique(age_bins):
        idx = (age_bins == bin)
        bin_mae.append(np.mean(error[idx]))
    bin_mae = np.asarray(bin_mae)

    return mae, bin_mae