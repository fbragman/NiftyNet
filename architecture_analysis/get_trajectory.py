from architecture_analysis.get_learned_categoricals import get_learned_categoricals
from architecture_analysis.simplex_viz import draw_trajectory
from architecture_analysis.simplex_viz import draw_heat_contours
import argparse
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib.pyplot as plt


def get_user_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path_to_mod", help="Path to models")
    parser.add_argument("-r", "--res", help="Save path")
    return parser.parse_args()


def get_checkpoint_files(model_path):

    p = Path(model_path)
    files = ['.'.join(str(x).split('.')) for x in p.glob('model.ckpt-*.meta')]
    files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
    return files


def create_heatmap_figure(cat, path_to_save, name):

    path_name_to_save = os.path.join(path_to_save, name + '_heatmap.pdf')
    fig = plt.figure(dpi=250, figsize=(7, 7))
    ax = fig.add_subplot(111)
    draw_heat_contours(cat, ax)
    fig.savefig(path_name_to_save)


def create_trajectory_figure(cats, path_to_save, name, it):

    path_name_to_save = os.path.join(path_to_save, name + '_trajectory.pdf')

    # Iterate over iteration
    layer_kernel_per_iter = []
    for iter, cat_iter in enumerate(cats):
        # Ger kernels for layer
        kernels = cat_iter[it]
        kernels_permuted = kernels[:, [1, 0, 2]]
        layer_kernel_per_iter.append(kernels_permuted)

    num_kernel = len(layer_kernel_per_iter[0])
    kernel_tuples = []
    for kernel_it in range(0, num_kernel):
        kernel_over_iter = [tuple(x[kernel_it]) for x in layer_kernel_per_iter]
        kernel_tuples.append(kernel_over_iter)

    fig = plt.figure(dpi=500, figsize=(7, 7))
    ax = fig.add_subplot(111)
    draw_trajectory(kernel_tuples, ax)

    fig.savefig(path_name_to_save)


def main(path_to_mod, path_to_save):

    # get checkpoint numbers in path_to_mod
    checkpoint_files = get_checkpoint_files(path_to_mod)

    cats_list = []
    for checkpoint_file in [checkpoint_files[-1]]:
        tmp_file = checkpoint_file.split('.meta')[0]
        print('loading {}'.format(checkpoint_file))
        cats, cat_names = get_learned_categoricals(tmp_file)
        cats_list.append(cats)

    # for the final iteration
    create_heatmap_figure(cats_list[0][0], path_to_save, 'layer_1')
    #create_heatmap_figure(cats_list[-1][0], path_to_save, 'layer_1')
    #create_heatmap_figure(cats_list[-1][1], path_to_save, 'layer_2')
    #create_heatmap_figure(cats_list[-1][2], path_to_save, 'layer_3')
    #create_heatmap_figure(cats_list[-1][3], path_to_save, 'layer_4')
    #create_heatmap_figure(cats_list[-1][4], path_to_save, 'layer_5')

    # for each layer - create image of learning trajectory
    create_trajectory_figure(cats_list, path_to_save, 'layer_1', 0)
    create_trajectory_figure(cats_list, path_to_save, 'layer_2', 1)
    create_trajectory_figure(cats_list, path_to_save, 'layer_3', 2)
    create_trajectory_figure(cats_list, path_to_save, 'layer_4', 3)
    create_trajectory_figure(cats_list, path_to_save, 'layer_5', 4)


if __name__ == "__main__":

    #args = get_user_params()

    res = '/scratch2/NOT_BACKED_UP/fbragman/DeepSyn/experiments/midl2019_experiments/with_dice_x_ent/model_2/hold_0'
    path_to_mod = os.path.join(res, 'models')

    main(path_to_mod, res)
