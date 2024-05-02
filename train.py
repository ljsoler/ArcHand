from utils import HandImagesDataModule
from utils import TrainManager
from utils import CallBackVerification
from datetime import datetime
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import time
import argparse
from lightning.pytorch.loggers import CSVLogger


def main(args):
    print('[INFO] Preparing directories')
    run_id = datetime.now().strftime('%m%d%H%M')
    run_dir = os.path.join(args.output_path, '{}_{}'.format(args.backbone, args.num_features))
    # os.makedirs(os.path.join(args.root_path, args.runs_dir), exist_ok=True)
    # os.makedirs(run_dir, exist_ok=True)

    print('[INFO] Configuring data module')
    data_module = HandImagesDataModule(
        data_path=args.data_dir,
        mask_dir=args.mask_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        input_size=args.input_size,
        hand=args.hand,
        use_mask=args.use_mask
    )

    data_module.setup()
    train_size = data_module.get_train_size()
    print(f'[INFO] Train size: {train_size}')
    train_templates_count = data_module.get_train_class_count()
    print(f'[INFO] Train class count: {train_templates_count}')
    val_size = data_module.get_val_size()
    print(f'[INFO] Validation size: {val_size}')
    val_templates_count = data_module.get_val_class_count()
    print(f'[INFO] Validation class count: {val_templates_count}')

    print('[INFO] Configuring model')
    model = TrainManager(model=args.backbone, num_features = args.num_features, num_identities=train_templates_count, s = args.S, m = args.M)

    # Initialise callbacks
    callbackVerification = CallBackVerification(data_module)
    monitor = 'EER_VER'
    filename = '{epoch}-{EER_VER:.2f}'
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor,
        dirpath=run_dir,
        filename=filename
    )

    logger = CSVLogger(run_dir, name="metrics")

    callbacks = [callbackVerification, checkpoint_callback]

    print('[INFO] Training model')
    start_time = time.time()
    trainer = pl.Trainer(max_epochs=args.max_epochs, 
                         check_val_every_n_epoch = args.val_freq, 
                         logger=logger,
                         callbacks=callbacks, 
                         log_every_n_steps=1)
    trainer.fit(model, data_module)
    end_time = time.time()
    training_time = time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))

    print('[INFO] Saving model')
    trainer.save_checkpoint(os.path.join(run_dir, f'model_{run_id}.ckpt'))

    with open(os.path.join(run_dir, f'run_{run_id}.log'), 'w') as f:
        f.write(f'[RUN INFO]\n')
        f.write(f'Run ID: {run_id}\n')
        f.write(f'Training time: {training_time}\n')
        f.write(f'Number of epochs: {args.max_epochs}\n')
        f.write(f'\n')
        f.write(f'[DATA INFO]\n')
        f.write(f'Data folder: {args.data_dir}\n')
        f.write(f'Train size: {train_size}\n')
        f.write(f'Train class count: {train_templates_count}\n')
        f.write(f'Validation size: {val_size}\n')
        f.write(f'Validation class count: {val_templates_count}\n')
        f.write(f'\n')
        f.write(f'[MODEL SETTINGS]\n')
        f.write(f'Image size: {args.input_size}x{args.input_size}\n')
        f.write(f'Batch size: {args.batch_size}\n')
        f.write(f'Number of workers: {args.num_workers}\n')
        f.write(f'Validation split: {args.val_split}\n')
        f.write(f'S: {args.S}\n')
        f.write(f'M: {args.M}\n')
        f.write(f'backbone: {args.backbone}\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch tattoo transformer network (TTN) for tattoo retrieval')
    parser.add_argument('--data_dir', type=str, help="path to database")
    parser.add_argument('--mask_dir', type=str, help="segmentation map dir")
    parser.add_argument('--output_path', type=str, help="output dir")
    parser.add_argument('--val_freq', type=int, default=5, help="frequency of validation")
    parser.add_argument('--num_features', type=int, default=512, help="embedding size")
    parser.add_argument('--max_epochs', type=int, default=30, help="max number of epochs")
    parser.add_argument('--input_size', type=int, default=324, help="image size to resize")
    parser.add_argument('--batch_size', type=int, default=128, help="batch size")
    parser.add_argument('--num_workers', type=int, default=3, help="number of workers to load the database")
    parser.add_argument('--val_split', type=float, default=0.05, help="rate to build validation database")
    parser.add_argument('--S', type=int, default=64, help="parameter used by the Angular Margin")
    parser.add_argument('--M', type=float, default=0.5, help="margin used by the Angular Margin")
    parser.add_argument('--hand', type=str, default='right',
                        choices=['right', 'left'],
                        help="gesture to train or to evaluate")
    parser.add_argument('--use_mask', action='store_true', default=False,
                    help='use the segmentation map of hands')
    parser.add_argument('--backbone', type=str, default='mobilenet_v3_large',
                        choices=['mba_net', 'mobilenet_v3_large', 'resnet101', 'densenet121', 'efficientnet_v2_s', 'swin_s'],
                        help="backbones to use")

    args_ = parser.parse_args()
    print(args_)
    main(args_)