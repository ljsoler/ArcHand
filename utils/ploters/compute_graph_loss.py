import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import argparse


'''
Prerequisites:
    - copy metrics.csv from lightning_logs to run folder
    - install matplotlib-3.8.2
'''



def create_loss_graph(run_file, result_dir):
    # Load the metrics CSV file
    df = pd.read_csv(run_file)

    # Create a new figure
    plt.figure()

    template_df = df[df['template_loss'].notna()]
    train_df = df[df['train_loss'].notna()]
    feature_df = df[df['features_loss'].notna()]

    # Plot the training loss
    plt.plot(train_df['epoch'], train_df['train_loss'], label='Train Loss')
    plt.plot(template_df['epoch'], template_df['template_loss'], label='Template Loss')
    plt.plot(feature_df['epoch'], feature_df['features_loss'], label='Features Loss')

    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save the figure
    plt.savefig(os.path.join(result_dir, 'loss_graph.png'))

    # Create a new figure
    plt.figure()

    fmr20_df = df[df['fmr20'].notna()]
    fmr100_df = df[df['fmr100'].notna()]
    eer_df = df[df['eer'].notna()]
    fmr10_df = df[df['fmr10'].notna()]
    reconstruction_df = df[df['reconstruction_loss'].notna()]

    # Plot the training loss
    plt.plot(fmr10_df['epoch'], fmr10_df['fmr10'], label='FMR10')
    plt.plot(fmr20_df['epoch'], fmr20_df['fmr20'], label='FMR20')
    plt.plot(fmr100_df['epoch'], fmr100_df['fmr100'], label='FMR100')
    plt.plot(eer_df['epoch'], eer_df['eer'], label='EER')
    plt.plot(reconstruction_df['epoch'], reconstruction_df['reconstruction_loss'], label='Reconstruction Loss')

    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()

    plt.savefig(os.path.join(result_dir, 'error_graph.jpg'))


def main(args):
    os.makedirs(args.output_folder, exist_ok=True)
    create_loss_graph(args.root_file, args.output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for generating training results')
    parser.add_argument('--root_file', type=str, help="root dir")
    parser.add_argument('--output_folder', type=str, help="name of the dir where runs are stored")

    args_ = parser.parse_args()
    print(args_)
    main(args_)