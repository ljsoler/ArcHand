import os
from pyeer.cmc_stats import load_scores_from_file, get_cmc_curve, CMCstats
from pyeer.plot import plot_cmc_stats
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def draw_cmc_plot(cmc_df, output_folder="cmc", FOLDS = 10):
    """
    Function making the figure of all cmc values

    Args:
        output_folder (str): output path

    """
    plt.figure(figsize=(8, 6))

    models = ['MobileNetv3', 'ResNet101', 'DenseNet121', 'EfficientNetv2', 'Swin']

    display = {
        "MobileNetv3": {"name": "MobileNetv3", "color": "green", "marker": 'x'},
        "ResNet101": {"name": "ResNet101", "color": "blue", "marker": 'x'},
        "DenseNet121": {"name": "DenseNet121", "color": "orange", "marker": 'x'},
        "EfficientNetv2": {"name": "EfficientNetv2", "color": "red", "marker": 'x'},
        "Swin": {"name": "Swin", "color": "black", "marker": 'x'}
}

    # Again go through all the model+experimets
    for model in models:
        model_name = model

        # For each model+experiment, we select the cmc values assosiated with it
        cmc_model = cmc_df[cmc_df["model"] == model_name]
        for c in cmc_model['accuracy']:
            print(c/100)
        # Then we plot the cmc values using errorbars. Usually we have multiple folds in cmc_model, i.e. we have multiple values for each rank.
        # Using sns.lineplot with errorbars automaticly computes the mean and std of each rank accuracy and plots it.
        sns.lineplot(data=cmc_model, x="rank", y="accuracy",
                        errorbar='sd', markersize=3, label=display[model_name]["name"],
                        color=display[model_name]["color"], marker=display[model_name]["marker"])

        # We also want to print the exact mean rank accuracy values, which is done by grouping and taking the mean
        print("{0}-fold mean rank of {1}".format(FOLDS, model_name))
        print(cmc_model[["rank", "accuracy"]].groupby(["rank"]).mean())

    # For making the figures nice to look at
    plt.legend(loc="lower right", fontsize=13)
    plt.xticks([1, 5, 10, 15, 20], fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(linestyle='--', linewidth=0.3)
    plt.xlabel("Rank", fontsize=18)
    plt.ylabel("Identification Rate (%)", fontsize=18)
    # plt.title("{0}-fold Cumlative Matching \n Characteristics (CMC) Curve".format(
    #     FOLDS), fontsize=23)

    output_path = os.path.join(output_folder, "cmc_plot.png")
    plt.savefig(output_path)


def main(args):
    cmc_df = pd.DataFrame(columns=["rank", "accuracy", "model"])

    for f in range(args.fold):
        # combined
        scores_path_comb = os.path.join(args.mobilenet_scores_dir, 'close_set_scores_{}.txt'.format(f))
        tp_path_comb = os.path.join(args.mobilenet_scores_dir, 'close_set_scores_tp_{}.txt'.format(f))
        scores_comb = load_scores_from_file(scores_path_comb, tp_path_comb)
        ranks_comb = get_cmc_curve(scores_comb, args.r)
        cmc = ranks_comb[:args.r]
        # Append cmc results to dataframe, stating the ranks, accuracy and model+experiment name
        cmc_df = cmc_df.append(pd.DataFrame(
            {"rank": range(1, args.r+1), "accuracy": np.array(cmc) * 100, "model": "MobileNetv3"}), ignore_index=True)

        # top
        scores_path_top = os.path.join(args.resnet_scores_dir, 'close_set_scores_{}.txt'.format(f))
        tp_path_top = os.path.join(args.resnet_scores_dir, 'close_set_scores_tp_{}.txt'.format(f))
        scores_top = load_scores_from_file(scores_path_top, tp_path_top)
        ranks_top = get_cmc_curve(scores_top, args.r)
        cmc = ranks_top[:args.r]
        # Append cmc results to dataframe, stating the ranks, accuracy and model+experiment name
        cmc_df = cmc_df.append(pd.DataFrame(
            {"rank": range(1, args.r+1), "accuracy": np.array(cmc) * 100, "model": "ResNet101"}), ignore_index=True)

        # bot
        scores_path_bot = os.path.join(args.densenet_scores_dir, 'close_set_scores_{}.txt'.format(f))
        tp_path_bot = os.path.join(args.densenet_scores_dir, 'close_set_scores_tp_{}.txt'.format(f))
        scores_bot = load_scores_from_file(scores_path_bot, tp_path_bot)
        ranks_bot = get_cmc_curve(scores_bot, args.r)
        cmc = ranks_bot[:args.r]
        # Append cmc results to dataframe, stating the ranks, accuracy and model+experiment name
        cmc_df = cmc_df.append(pd.DataFrame(
            {"rank": range(1, args.r+1), "accuracy": np.array(cmc) * 100, "model": "DenseNet121"}), ignore_index=True)
        
        # bot
        scores_path_bot = os.path.join(args.efficientnet_scores_dir, 'close_set_scores_{}.txt'.format(f))
        tp_path_bot = os.path.join(args.efficientnet_scores_dir, 'close_set_scores_tp_{}.txt'.format(f))
        scores_bot = load_scores_from_file(scores_path_bot, tp_path_bot)
        ranks_bot = get_cmc_curve(scores_bot, args.r)
        cmc = ranks_bot[:args.r]
        # Append cmc results to dataframe, stating the ranks, accuracy and model+experiment name
        cmc_df = cmc_df.append(pd.DataFrame(
            {"rank": range(1, args.r+1), "accuracy": np.array(cmc) * 100, "model": "EfficientNetv2"}), ignore_index=True)
        
        # bot
        scores_path_bot = os.path.join(args.swin_scores_dir, 'close_set_scores_{}.txt'.format(f))
        tp_path_bot = os.path.join(args.swin_scores_dir, 'close_set_scores_tp_{}.txt'.format(f))
        scores_bot = load_scores_from_file(scores_path_bot, tp_path_bot)
        ranks_bot = get_cmc_curve(scores_bot, args.r)
        cmc = ranks_bot[:args.r]
        # Append cmc results to dataframe, stating the ranks, accuracy and model+experiment name
        cmc_df = cmc_df.append(pd.DataFrame(
            {"rank": range(1, args.r+1), "accuracy": np.array(cmc) * 100, "model": "Swin"}), ignore_index=True)

    # Stats
    plots_dir = os.path.join(args.root_dir, 'cmc')
    os.makedirs(plots_dir, exist_ok=True)

    draw_cmc_plot(cmc_df, plots_dir, args.r)

    # # Creating stats
    # stats = [CMCstats(exp_id='RIE', ranks=ranks_bot),
    #          CMCstats(exp_id='TE', ranks=ranks_top),
    #          CMCstats(exp_id='RIE+TE', ranks=ranks_comb)
    # ]

    # # Plotting
    # plot_cmc_stats(stats, args.r, save_path=plots_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for generating CMC curves for the top, bot and combined pipelines')
    parser.add_argument('--mobilenet_scores_dir', type=str, 
                        default='/Users/soler/Research_Projects/Others-Projects/Github-projects/tattoo-retrieval/models/mobilenet_v3_large_4.0_0.5_512/03040959/scores',
                        help="scores directory for combined pipeline")
    parser.add_argument('--resnet_scores_dir', type=str, 
                        default='/Users/soler/Research_Projects/Others-Projects/Github-projects/tattoo-retrieval/models/resnet101_4.0_0.5_512/03041024/scores',
                        help="scores directory for top pipeline")
    parser.add_argument('--densenet_scores_dir', type=str, 
                        default='/Users/soler/Research_Projects/Others-Projects/Github-projects/tattoo-retrieval/models/densenet121_4.0_0.1_256/03041434/scores',
                        help="scores directory for bot pipeline")
    parser.add_argument('--efficientnet_scores_dir', type=str, 
                        default='/Users/soler/Research_Projects/Others-Projects/Github-projects/tattoo-retrieval/models/efficientnet_v2_s_4.0_0.1_512/03041042/scores',
                        help="scores directory for bot pipeline")
    parser.add_argument('--swin_scores_dir', type=str, 
                        default='/Users/soler/Research_Projects/Others-Projects/Github-projects/tattoo-retrieval/models/swin_s_4.0_0.1_512/03041506/scores',
                        help="scores directory for bot pipeline")
    parser.add_argument('--root_dir', type=str, 
                        default='/Users/soler/Research_Projects/Others-Projects/Github-projects/tattoo-retrieval/results/WebTattoo_plots',
                        help="root dir to create plots folder for")
    parser.add_argument('--r', type=int, default=20, help="ranks")
    parser.add_argument('--fold', type=int, default=10, help="fold number to plot the results for")


    args_ = parser.parse_args()
    print(args_)
    main(args_)
