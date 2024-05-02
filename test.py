from utils.dataset import HaGridDB
from utils import TrainManager
from utils import CallBackVerification, CallBackSimilarityExtraction
from datetime import datetime
import os
import pytorch_lightning as pl
import time
import argparse
from torchvision import transforms
import torch
import csv


def main(args):
    print('[INFO] Preparing directories')
    scores_dir = os.path.join(args.output_path)
    os.makedirs(scores_dir, exist_ok=True)

    test_transform = transforms.Compose([
                    transforms.Resize((args.input_size, args.input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    print('[INFO] Configuring data module')
    data_test = HaGridDB(args.data_dir, args.mask_dir, isTraining=False, hand=args.hand, gesture=args.gesture, use_masks=args.use_mask, transform=test_transform)

    test_dataloader = torch.utils.data.DataLoader(data_test,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          num_workers=args.num_workers
                                        )

    
    # print("Len of test_dataloader: " + str(len(test_dataloader)))

    model = TrainManager(model=args.backbone, num_features = args.num_features, num_identities=11670, s = args.S, m = args.M)

    # Initialise testing callbacks
    callbackVerification = CallBackSimilarityExtraction(test_dataloader, scores_dir)

    print('[INFO] Testing model')
    start_time = time.time()
    trainer = pl.Trainer(max_epochs=args.max_epochs, callbacks=[
        callbackVerification
    ])
    results = trainer.test(model, dataloaders=test_dataloader, ckpt_path=args.weights)
    # results[0]['NoId'] = str(data_test.get_num_identities())
    results[0]['NoId'] = str(len(set(data_test.classes)))
    end_time = time.time()
    training_time = time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))

    with open(os.path.join(scores_dir, 'metric_info_{}.csv'.format(args.gesture)), mode='w') as csv_file:
        fieldnames = ['NoId', 'EER', 'FMR10', 'FMR20', 'FMR100']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(results[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch tattoo transformer network (TTN) for tattoo retrieval')
    parser.add_argument('--data_dir', type=str, help="path to database")
    parser.add_argument('--mask_dir', type=str, help="segmentation map dir")
    parser.add_argument('--output_path', type=str, help="output dir")
    parser.add_argument('--weights', type=str, help="model's weights")
    parser.add_argument('--val_freq', type=int, default=5, help="frequency of validation")
    parser.add_argument('--num_features', type=int, default=512, help="embedding size")
    parser.add_argument('--max_epochs', type=int, default=30, help="max number of epochs")
    parser.add_argument('--input_size', type=int, default=324, help="image size to resize")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--num_workers', type=int, default=2, help="number of workers to load the database")
    parser.add_argument('--val_split', type=float, default=0.05, help="rate to build validation database")
    parser.add_argument('--S', type=int, default=64, help="parameter used by the Angular Margin")
    parser.add_argument('--M', type=float, default=0.5, help="margin used by the Angular Margin")
    parser.add_argument('--hand', type=str, default='right',
                        choices=['right', 'left'],
                        help="hand side to train or to evaluate")
    parser.add_argument('--gesture', type=str, default='palm',
                        choices=['call', 'dislike', 'fist', 'four', 'like', 'mute', 'ok', 'one', 'palm', 'peace', 'peace_inverted', 'rock', 'stop', 'stop_inverted', 'three', 'three2', 'two_up', 'two_up_inverted', 'None'],
                        help="gesture to train or to evaluate")
    parser.add_argument('--use_mask', action='store_true', default=False,
                    help='use the segmentation map of hands')
    parser.add_argument('--backbone', type=str, default='mobilenet_v3_large',
                        choices=['mba_net', 'mobilenet_v3_large', 'resnet101', 'densenet121', 'efficientnet_v2_s', 'swin_s'],
                        help="backbones to use")

    args_ = parser.parse_args()
    print(args_)
    main(args_)