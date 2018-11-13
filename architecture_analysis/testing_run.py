import os
import os
from architecture_analysis.simplex_viz import draw_pdf_contours
from architecture_analysis.get_learned_categoricals import get_learned_categoricals

def main(checkpoint_path, output_path):

    cats, cat_names = get_learned_categoricals(checkpoint_path, output_path)

    for cat, cat_name in zip(cats, cat_names):
        cat_layer_layer = '_'.join(cat_name.split('/')[2].split('_')[:2])
        fig_save = os.path.join(output_path, cat_layer_layer + '.png')
        draw_pdf_contours(cat, fig_save)

if __name__ == '__main__':

    path_to_model = '/scratch2/NOT_BACKED_UP/fbragman/DeepSyn/experiments/age_gender/baseline_root/baseline_5_mt/soft_gumbel_softmax_no_entropy/model_0/vgg16/models'
    path_to_checkpoint = os.path.join(path_to_model, 'model.ckpt-50000')
    output_dir = '/scratch2/NOT_BACKED_UP/fbragman/DeepSyn/experiments/age_gender/baseline_root/baseline_5_mt/soft_gumbel_softmax_no_entropy/model_0/vgg16/categorical_output'

    main(path_to_checkpoint, output_dir)


