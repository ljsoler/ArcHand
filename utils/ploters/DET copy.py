import os
import numpy as np
from DET.DET import DET
import argparse

'''
Prerequisites:
    - install matplotlib-3.3.2
'''


def main(args):
    systems = {}

    data_mobilenet = np.load(os.path.join(args.mobilenet_scores_dir, 'open_set_scores.npz'))
    systems['MobileNetv3'] = {}
    systems['MobileNetv3']['mated'] = data_mobilenet['gen']
    systems['MobileNetv3']['non-mated'] = data_mobilenet['imp']
    systems['MobileNetv3']['label'] = 'MobileNetv3'

    data_ResNet101 = np.load(os.path.join(args.resnet_scores_dir, 'open_set_scores.npz'))
    systems['ResNet101'] = {}
    systems['ResNet101']['mated'] = data_ResNet101['gen']
    systems['ResNet101']['non-mated'] = data_ResNet101['imp']
    systems['ResNet101']['label'] = 'ResNet101'

    data_DenseNet121 = np.load(os.path.join(args.densenet_scores_dir, 'open_set_scores.npz'))
    systems['DenseNet121'] = {}
    systems['DenseNet121']['mated'] = data_DenseNet121['gen']
    systems['DenseNet121']['non-mated'] = data_DenseNet121['imp']
    systems['DenseNet121']['label'] = 'DenseNet121'

    data_EfficientNetv2 = np.load(os.path.join(args.efficientnet_scores_dir, 'open_set_scores.npz'))
    systems['EfficientNetv2'] = {}
    systems['EfficientNetv2']['mated'] = data_EfficientNetv2['gen']
    systems['EfficientNetv2']['non-mated'] = data_EfficientNetv2['imp']
    systems['EfficientNetv2']['label'] = 'EfficientNetv2'

    data_Swin = np.load(os.path.join(args.swin_scores_dir, 'open_set_scores.npz'))
    systems['Swin'] = {}
    systems['Swin']['mated'] = data_Swin['gen']
    systems['Swin']['non-mated'] = data_Swin['imp']
    systems['Swin']['label'] = 'Swin'

    det = DET(biometric_evaluation_type='identification', abbreviate_axes=True,
              plot_eer_line=True)
    det.x_limits = np.array([1e-4, .9])
    det.y_limits = np.array([1e-4, .9])
    det.x_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
    det.x_ticklabels = np.array(['0.1', '1', '5', '20', '40'])
    det.y_ticks = np.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
    det.y_ticklabels = np.array(['0.1', '1', '5', '20', '40'])

    det.create_figure()

    color = ['green', 'blue', 'orange', 'red', 'black']

    col = 0
    for system in systems.keys():
        mated = systems[system]['mated']
        non_mated = systems[system]['non-mated']
        det.plot(tar=mated, non=non_mated, label=systems[system]['label'], plot_rocch=True,
                 plot_args=(color[col], '-', '1.5'))
        col += 1

    det.legend_on(loc='upper right', fontsize=12)

    plots_dir = os.path.join(args.root_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    det.save(os.path.join(plots_dir, 'DET'), 'png')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for generating DET curves for the top, bot and combined pipelines')
    parser.add_argument('--mobilenet_scores_dir', type=str, help="scores directory for combined pipeline",
                        default='/Users/soler/Research_Projects/Others-Projects/Github-projects/tattoo-retrieval/models/mobilenet_v3_large_4.0_0.5_512/03040959/scores')
    parser.add_argument('--resnet_scores_dir', type=str, help="scores directory for top pipeline",
                        default='/Users/soler/Research_Projects/Others-Projects/Github-projects/tattoo-retrieval/models/resnet101_4.0_0.5_512/03041024/scores')
    parser.add_argument('--densenet_scores_dir', type=str, help="scores directory for bot pipeline",
                        default='/Users/soler/Research_Projects/Others-Projects/Github-projects/tattoo-retrieval/models/densenet121_4.0_0.1_256/03041434/scores')
    parser.add_argument('--efficientnet_scores_dir', type=str, help="scores directory for bot pipeline",
                        default='/Users/soler/Research_Projects/Others-Projects/Github-projects/tattoo-retrieval/models/efficientnet_v2_s_4.0_0.1_512/03041042/scores')
    parser.add_argument('--swin_scores_dir', type=str, help="scores directory for bot pipeline",
                        default='/Users/soler/Research_Projects/Others-Projects/Github-projects/tattoo-retrieval/models/swin_s_4.0_0.1_512/03041506/scores')
    parser.add_argument('--root_dir', type=str, help="root dir to create plots folder for",
                        default='/Users/soler/Research_Projects/Others-Projects/Github-projects/tattoo-retrieval/results/WebTattoo_plots')

    args_ = parser.parse_args()
    print(args_)
    main(args_)