import os
import subprocess


def main():

    root = os.path.dirname(os.path.realpath(__file__))
    print('Operating in path {}'.format(root))
    config = os.path.join(root, 'config.ini')

    application = '/scratch2/NOT_BACKED_UP/fbragman/DeepSyn/code/NiftyNet_github_fork/NiftyNet/net_multitask.py'

    special_python = '/home/fbragman/miniconda3/envs/tf_d/bin/python'

    subprocess.call([special_python, application, 'train', '-c' + config])


if __name__ == '__main__':
    main()
