import numpy as np
import nibabel as nib


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


def get_prediction(files):
    """
    Load image and get prediction
    :param files:
    :return:
    """
    predictions = []
    for f in files:
        tmp = nib.load(str(f)).get_data()[0].squeeze()
        predictions.append(tmp)
    return np.asarray(predictions)


def calculate_mae(files):
    """
    Calculate mean absolute error
    :param files:
    :return:
    """
    gt = get_ground_truth(files, 'age')
    pred = get_prediction(files)
    # age * 100
    pred = pred * 100
    # calculate mean absolute error
    mae = np.mean(np.abs(pred-gt))

    # get binned results