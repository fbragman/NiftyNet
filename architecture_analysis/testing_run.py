import os
from architecture_analysis.get_learned_categoricals import plot_learned_categoricals

def main(checkpoint_path, output_path):

    plot_learned_categoricals(checkpoint_path, output_path)


if __name__ == '__main__':

    path_to_model = '''
    /scratch2/NOT_BACKED_UP/fbragman/DeepSyn/experiments/age_gender/baseline_root/baseline_5_mt/
    soft_gumbel_softmax_no_entropy/model_0/vgg16/models
    '''
    path_to_checkpoint = os.path.join(path_to_model, 'model.ckpt-50000.meta')

    output_dir = '''/scratch2/NOT_BACKED_UP/fbragman/DeepSyn/experiments/age_gender/baseline_root/baseline_5_mt/
    soft_gumbel_softmax_no_entropy/model_0/vgg16/categorical_output
    '''

    main(path_to_checkpoint, output_dir)


