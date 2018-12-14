from architecture_analysis.get_learned_categoricals import get_learned_categoricals
from architecture_analysis.simplex_viz import draw_pdf_contours
import argparse
from pathlib import Path
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as manimation
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def get_user_params():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path_to_mod", help="Path to models")
    parser.add_argument("-r", "--res", help="Save path")
    return parser.parse_args()


def get_checkpoint_files(path_to_mod):

    p = Path(path_to_mod)
    files = ['.'.join(str(x).split('.')[0:2]) for x in p.glob('model.ckpt-*.meta')]
    files.sort(key=lambda x: int(x.split('-')[-1]))
    return files


def create_video(cats, path_to_save, name, it):

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=5, metadata=metadata)

    fig = plt.figure(dpi=500, figsize=(5, 5))
    path_name_to_save = os.path.join(path_to_save, name + '.mp4')

    with writer.saving(fig, path_name_to_save, len(cats)):
        for cat in cats:
            l_cat = cat[it]
            p_cat = l_cat[:, [1, 0, 2]]
            ax = fig.add_subplot(111)
            draw_pdf_contours(p_cat, ax)
            fig.canvas.draw()
            writer.grab_frame()


def main(path_to_mod, path_to_save):

    # get checkpoint numbers in path_to_mod
    checkpoint_files = get_checkpoint_files(path_to_mod)

    cats_list = []
    for checkpoint_file in checkpoint_files:
        print('loading {}'.format(checkpoint_file))
        cats, cat_names = get_learned_categoricals(checkpoint_file)
        cats_list.append(cats)

    # for each layer - create video of learning
    #create_video(cats_list, path_to_save, 'layer_1', 0)
    #create_video(cats_list, path_to_save, 'layer_2', 1)
    #create_video(cats_list, path_to_save, 'layer_3', 2)
    #create_video(cats_list, path_to_save, 'layer_4', 3)
    create_video(cats_list, path_to_save, 'layer_5', 4)

if __name__ == "__main__":

    #args = get_user_params()

    path_to_mod = '/scratch2/NOT_BACKED_UP/fbragman/DeepSyn/experiments/cvpr2018_experiments/baseline_5/anneal_decay_analysis/gpu_0_runs/model_0_0/vgg16/models'
    res = '/scratch2/NOT_BACKED_UP/fbragman/DeepSyn/experiments/cvpr2018_experiments/baseline_5/anneal_decay_analysis/gpu_0_runs/model_0_0/vgg16'

    main(path_to_mod, res)
