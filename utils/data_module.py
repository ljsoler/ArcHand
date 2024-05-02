import random
from collections import defaultdict
import numpy as np
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from utils.dataset import HaGridDB
from torchvision import transforms


class HandImagesDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str, mask_dir: str, batch_size: int, num_workers: int, 
                 val_split: float, input_size: int, hand: str, use_mask:bool = False):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.input_size = input_size
        self.mask_dir = mask_dir
        self.use_mask = use_mask
        self.hand = hand

        #input transformers
        self.transform = transforms.Compose([
                    transforms.Resize((input_size, input_size)),
                    transforms.ColorJitter(brightness=(0.75,1.25),contrast=(1),saturation=(0.75,1.25)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        self.transform_val = transforms.Compose([
                    transforms.Resize((input_size, input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        
    def setup(self, stage=None):
        if hasattr(self, "train_data") and hasattr(self, "val_data"):
            return

        self.train_data = HaGridDB(self.data_path, self.mask_dir, isTraining=True, hand=self.hand, use_masks=self.use_mask, transform=self.transform)
        self.val_data = HaGridDB(self.data_path, self.mask_dir, isTraining=False, hand=self.hand, use_masks=self.use_mask, transform=self.transform_val)
        
        # print("[INFO] Train len -> {}".format(len(self.train_data.labels)))
        # print("[INFO] Test len -> {}".format(len(self.val_data.labels)))

        self.train_templates = max(self.train_data.labels) + 1
        self.val_templates = max(self.val_data.labels) + 1

        self.t_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, shuffle=True)
        self.v_dataloader = DataLoader(self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def train_dataloader(self):
        return self.t_dataloader

    def val_dataloader(self):
        return self.v_dataloader

    def get_train_size(self):
        return len(self.train_data)

    def get_val_size(self):
        return len(self.val_data)

    def get_train_class_count(self):
        return self.train_templates

    def get_val_class_count(self):
        return self.val_templates

