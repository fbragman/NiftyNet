import argparse
import os
import pandas as pd
from evaluation import get_prediction_parallel
import time
import numpy as np
from scipy import stats

from sklearn.metrics import accuracy_score


def get_user_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path_to_res", help="Path to inference results.")
    parser.add_argument("-t", "--num_tasks", help="Number of tasks")
    parser.add_argument("-t1", "--task_1", help="Number of tasks")
    parser.add_argument("-t2", "--task_2", help="Number of tasks")
    return parser.parse_args()


def calculate_mae(results):
    mae = []
    std = []
    for res in results:
        gt = res[0]
        gt_val = float(gt.split('_')[0])
        vals = res[1:]
        # destroy -69.69 if exist
        vals = np.delete(vals, np.argwhere(vals == -69.69))
        # If network prediction is -ive, set to age = 0
        vals[vals<0] = 0
        mean_val = np.mean(vals) * 100
        std_val = np.std(vals) * 100
        mae.append(np.abs(gt_val - mean_val))
        std.append(std_val)

    return np.mean(mae), np.mean(std)


def calculate_reg_mode(results):

    mae = []
    it = 0
    for res in results:
        gt = res[0]
        gt_val = float(gt.split('_')[0])
        vals = res[1:]
        # destroy -69.69 if exist
        vals = np.delete(vals, np.argwhere(vals == -69.69))
        x = np.linspace(np.min(vals), np.max(vals), 1000)
        kde = stats.kde.gaussian_kde(vals.astype(np.float32), bw_method=2.05)
        #fig, ax = plt.subplots(1, 1)
        #ax.plot(x, kde(x))
        mode = x[np.argsort(kde(x))[-1]]
        mode_val = 100*mode
        #print('Mode: {} Mean: {}, GT: {}'.format(100*mode, 100*np.mean(vals), gt_val))
        mae.append(np.abs(gt_val - mode_val))
        #plt.savefig('/home/fbragman/documents/tmp/fig_{}.png'.format(it))
        it += 1

    return np.mean(mae), None


def calculate_mode(results):
    mode = []
    gts = []
    for res in results:
        gt = res[0]
        gts.append(int(gt.split('_')[1]))
        vals = res[1:]
        # destroy -69.69 if exist
        vals = np.delete(vals, np.argwhere(vals == -69.69))
        mode.append(stats.mode(vals)[0][0])

    return accuracy_score(gts, mode, 'weighted')


def main(path_to_res, num_tasks, t1, t2):

    # get list of directories called output_iter_***
    dirs = os.listdir(path_to_res)
    dirs = [x for x in dirs if 'output_iter_' in x]
    dirs.sort()

    if num_tasks == 1:
        task_1_res = {}
    else:
        task_1_res = {}
        task_2_res = {}

    if t1 is not None:
        path_to_res_1 = os.path.join(path_to_res, t1 + '.csv')
        type_1 = t1.split('_')[1]
        print(path_to_res_1)
        print(type_1)

    if t2 is not None:
        path_to_res_2 = os.path.join(path_to_res, t2 + '.csv')
        type_2 = t2.split('_')[1]
    else:
        path_to_res_2 = None

    if os.path.exists(path_to_res_1):

        res_1 = pd.read_csv(path_to_res_1, header=None)
        if type_1 == 'regression':
            reg_results = calculate_mae(res_1.values)
            print('MAE: {} std: {}'.format(reg_results[0], reg_results[1]))
        else:
            acc_result = calculate_mode(res_1.values)
            print('Accuracy: {}'.format(acc_result))

        if t2 is not None:
            res_2 = pd.read_csv(path_to_res_2, header=None)
            if type_2 == 'regression':
                reg_results = calculate_mae(res_2.values)
                print('MAE: {} std: {}'.format(reg_results[0], reg_results[1]))
            else:
                acc_result = calculate_mode(res_2.values)
                print('Accuracy: {}'.format(acc_result))

    else:

        # loop over results
        for dir_it, dir in enumerate(dirs):

            start = time.time()
            print('Getting results from {}'.format(dir))

            # root path
            inference_iter_path = os.path.join(path_to_res, dir)

            # loop over patients
            patient_results = os.listdir(inference_iter_path)
            patient_files = ['_'.join(x.split('_')[0:-2]) for x in patient_results]
            patient_uniq = list(set(patient_files))
            patient_uniq = [x for x in patient_uniq if x is not '']
            patient_uniq = [x.split('task')[0] for x in patient_uniq]

            # get task names
            task_names = ['_'.join(x.split('_')[4:]) for x in patient_results]
            task_names = list(set(task_names))
            task_names = [x for x in task_names if x is not '']
            task_names = [x for x in task_names if 'task' in x or 'niftynet' in x]
            #patient_uniq = ['_'.join(x.split('_')[:-1]) for x in patient_uniq]

            print(task_names)
            print('ahaha')
            #task_names = [x.split('_')[1] for x in task_names]

            # loop over tasks
            for task_it, task in enumerate(task_names):

                # create list of files
                patient_task_files = [os.path.join(inference_iter_path, '_'.join([str(x), task]))
                                      for x in patient_uniq]

                # get results
                task_results = get_prediction_parallel(patient_task_files)

                # update dictionary with results for patients
                if task_it == 0:
                    for pat_it, pat in enumerate(patient_uniq):
                        if pat in task_1_res:
                            task_1_res[pat].append(task_results[pat_it])
                        else:
                            task_1_res[pat] = [task_results[pat_it]]
                else:
                    for pat_it, pat in enumerate(patient_uniq):
                        if pat in task_2_res:
                            task_2_res[pat].append(task_results[pat_it])
                        else:
                            task_2_res[pat] = [task_results[pat_it]]

            end = time.time()
            print('Time per iter: {}'.format(end-start))

        # strip task_names
        tasks = [x.split('.')[0] for x in task_names]

        # Get number of values: should be 100...
        max_val = get_max_val(task_1_res)
        # Fill results with NaN in case..
        task_1_res = fill_with_nan(task_1_res, max_val)
        if len(tasks) > 1:
            task_2_res = fill_with_nan(task_2_res, max_val)

        # create pandas dataframe and save to csv
        for T_it, T in enumerate(tasks):
            if T_it == 0:
                task_1_res = convert_dict_to_list(task_1_res)
                task_1_pd = pd.DataFrame(task_1_res)
                task_1_save_path = os.path.join(path_to_res, T + '.csv')
                task_1_pd.to_csv(task_1_save_path, header=None, index=False)
            if T_it > 0:
                task_2_res = convert_dict_to_list(task_2_res)
                task_2_pd = pd.DataFrame(task_2_res)
                task_2_save_path = os.path.join(path_to_res, T + '.csv')
                task_2_pd.to_csv(task_2_save_path, header=None, index=False)


def fill_with_nan(di, max_val):
    for key, value in di.items():
        if len(value) < max_val:
            for _ in range(max_val-len(value)):
                value.append(-69.69)
            di[key] = value
    return di


def convert_dict_to_list(di):

    res = []
    for key, value in di.items():
        res.append([key] + value)
    return res


def get_max_val(res_dict):
    itr = 0
    max_val = 1
    for _, value in res_dict.items():
        if itr == 0:
            max_val = len(value)
        else:
            if len(value) > max_val:
                max_val = len(value)
        itr += 1
    return max_val


if __name__ == "__main__":

    args = get_user_params()

    if args.task_1 is not None:
        t1 = args.task_1
    else:
        t1 = None

    if args.task_2 is not None:
        t2 = args.task_2
    else:
        t2 = None

    main(args.path_to_res, int(args.num_tasks), t1, t2)
