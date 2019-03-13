import os
import scipy.ndimage.filters
from os.path import join
import numpy as np
import re
import matplotlib.pyplot as plt


def get_flags_to_look(task_type, model_type, recip_flag, reverse_flag):

    task_1 = None
    task_2 = None
    recip = [False, False]

    if task_type == 'multi':
        if model_type == 'face':
            task_1 = 'task_1_mae'
            task_2 = 'task_2_accuracy'
            recip = [True, False]
    else:
        if model_type == 'face':
            task_2 = None
            if recip_flag is True:
                task_1 = 'loss'
                recip = [True, False]
            else:
                task_1 = 'loss'
                recip = [False, False]

    if reverse_flag is True:
        task_1, task_2 = task_2, task_1
        task_1 = task_1.replace('2', '1')
        task_2 = task_2.replace('1', '2')
        recip[0], recip[1] = recip[1], recip[0]

    return task_1, task_2, recip


def analyse_validation_loss(path_to_log, path_to_checkpoints, task_type, model_type,
                            flag, reverse_flag):
    """

    :param path_to_log:
    :param path_to_checkpoints:
    :param task_type: single or multi
    :param model_type: face or prostate
    :param flag: True or False - if True: regression, if False: segmentation/classification
    :param reverse_flag: if True, it's because a random model had wrong order of tasks
    :return:
    """

    while True:
        log_file = join(path_to_log, 'training_niftynet_log')
        print(log_file)
        # get training data from console output
        validation_lines = get_validation_lines(log_file)

        task_1, task_2, recip = get_flags_to_look(task_type,
                                                  model_type,
                                                  recip_flag=flag,
                                                  reverse_flag=reverse_flag)

        iter = calculate_best_val_loss(validation_lines, task_1, task_2, path_to_log, recip, task_1, task_2)

        # find saved tensorflow checkpoints and saved iteration closest to global_iter
        closest_check_point_iter = search_checkpoints(path_to_checkpoints, iter)

        return closest_check_point_iter
    else:
        return False


def calculate_best_val_loss(val_lines, str_1, str_2, save_path, recip, xlabel, ylabel):

    # Calculate average per validation iteration
    iters = get_data(val_lines, 'iter')
    # Get unique
    unique_iters = np.unique(iters)

    str_1_values = np.asarray(get_data(val_lines, str_1))
    if str_2 is not None:
        str_2_values = np.asarray(get_data(val_lines, str_2))

    loss_val_1 = []
    loss_val_2 = []
    for val_iter in unique_iters:
        # get average for str_1 at all positions for iteration
        pos_of_val_iter = iters == val_iter
        tmp_avg_str_1 = np.mean(str_1_values[pos_of_val_iter])
        if str_2 is not None:
            tmp_avg_str_2 = np.mean(str_2_values[pos_of_val_iter])

        loss_val_1.append(tmp_avg_str_1)
        if str_2 is not None:
            loss_val_2.append(tmp_avg_str_2)
        else:
            loss_val_2.append(0)

    # Smooth both loss val
    filtered_1 = scipy.ndimage.filters.median_filter(loss_val_1, size=10)
    if str_2 is not None:
        filtered_2 = scipy.ndimage.filters.median_filter(loss_val_2, size=10)
    else:
        filtered_2 = loss_val_2

    if recip[0]:
        r_filtered_1 = 1-filtered_1
    else:
        r_filtered_1 = filtered_1

    if recip[1]:
        r_filtered_2 = 1-filtered_2
    else:
        r_filtered_2 = filtered_2

    both_added = r_filtered_2 + r_filtered_1
    max_idx = np.argmax(both_added)
    best_iter = unique_iters[max_idx]

    plot_validation_loss(unique_iters, loss_val_1, loss_val_2,
                         filtered_1, filtered_2, best_iter, save_path, xlabel, ylabel)

    return best_iter


def search_checkpoints(path_to_checkpoints, min_iter):

    # list index files
    index_files = [x for x in os.listdir(path_to_checkpoints) if 'index' in x]
    # extract numbers
    index_num = []
    for index_file in index_files:
        index_num.append(int(re.findall('\d+', index_file)[0]))
    index_num = np.unique(index_num)

    # find closest
    return index_num[np.argmin(np.abs(index_num-min_iter))]


def get_data(data, var):
    output = []
    for tmp in data:
        str_val_pos = tmp.find(var)
        tmp_data = get_numerical_value(tmp, str_val_pos, var)
        output.append(tmp_data)
    return output


def get_numerical_value(input_data, int_start, var):
    len_var = len(var) + 1
    start_search = int_start + len_var
    final_val = start_search

    for tmp_s in input_data[start_search:]:
        final_val += 1
        if tmp_s == ',' or tmp_s.isspace():
            # stop and get value
            final_val = final_val - 1
            result = float(input_data[start_search:final_val])
            return result


def get_validation_lines(log_file):
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            log = list(f)

        # find all validation lines
        new_validation_lines = [x for x in log if 'validation iter' in x]



        # since we re-ran stuff and they got saved in log file, only look for lines with task_1_accuracy and task_2_mae
        #if task_flag == 1:
        #    new_validation_lines = [x for x in validation_lines if 'task_2_mae' in x]
        #elif task_flag == 2:
        #    new_validation_lines = [x for x in validation_lines if 'task_2_mae' in x]
        #else:
        #    new_validation_lines = [x for x in validation_lines if 'loss' in x]

        # since we are stupid and we possibly stopped training and re-ran once, need to find re-start
        # get iterations and find index of first repeat
        val_iter = get_data(new_validation_lines, 'iter')
        # find iteration start in case of new logged runs
        val_idx = [x == val_iter[0] for x in val_iter]
        # find where start happened again
        val_pos = np.where(val_idx)
        # find consecutive ones since we perform n validation iteration per validation
        val_dif = np.ediff1d(val_pos[0]) != 1
        val_dif_not_one = np.where(val_dif)
        # get biggest index
        if val_dif_not_one[0].size == 0:
            val_start = 0
        else:
            val_start = np.max(val_dif_not_one[0]) + 1
        val_start_pos = val_pos[0][val_start]

        # validation lines to analyse
        res_validation_lines = new_validation_lines[val_start_pos:]

        return res_validation_lines
    else:
        print('log file not found')
        return False


def plot_validation_loss(x, y_1, y_2, z_1, z_2, best_iter, save_path, xlabel, ylabel):

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('global iteration')
    ax1.set_ylabel(xlabel, color=color)
    ax1.plot(x, y_1, color=color, alpha=0.5)
    ax1.plot(x, z_1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax1.axvline(x=best_iter, color='black')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel(ylabel, color=color)  # we already handled the x-label with ax1
    ax2.plot(x, y_2, color=color, alpha=0.5)
    ax2.plot(x, z_2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig_name = os.path.join(save_path, 'validation_performance.png')
    plt.savefig(fig_name)
    plt.close(fig)
