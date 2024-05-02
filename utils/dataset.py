from torchvision.datasets import ImageFolder
import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob
import numpy as np
import cv2
from PIL import Image


class HaGridDB(Dataset):
    def __init__(self, root_dir: str, mask_dir: str, isTraining=True, hand='right', gesture='palm', use_masks=False, transform=None):
        super(HaGridDB, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.mask_dir = mask_dir
        self.isTraining = isTraining
        self.use_masks = use_masks
        self.gesture = gesture
        self.hand = hand
        self.images, self.labels, self.id_per_sample, self.masks, self.correct_masks = self._scan()

    def _scan(self):
        labels = []
        images_path = []
        masks = []
        id_per_sample = []
        correct_masks = []

        input_path = os.path.join(self.root_dir, 'train' if self.isTraining else 'test')
        subject_id = list(os.listdir(input_path))
        id_count = 0
        for id in subject_id:
            
            regular_exp = '{}/{}/*/{}/*.jpg'.format(input_path, id, self.hand) if 'None' in self.gesture else '{}/{}/{}/{}/*.jpg'.format(input_path, id, self.gesture, self.hand)
            images = list(glob.glob(regular_exp, recursive=True))
            
            if len(images) > 1:
                labels = [*labels, *[id_count] * len(images)]
                images_path = [*images_path, *images.copy()]
                id_per_sample = [*id_per_sample, *[id] * len(images)]
                id_count += 1
                
                if self.use_masks:
                    masks_path = [os.path.join(self.mask_dir, '{}/{}/{}/{}/{}.png'.format('train' if self.isTraining else 'test', Path(f).parent.parent.parent.name, Path(f).parent.parent.name, Path(f).parent.name, Path(f).stem)) for f in images]
                    correct_masks_tmp = [np.count_nonzero(cv2.imread(f, cv2.IMREAD_GRAYSCALE)) for f in masks_path]
                    masks = [*masks, *masks_path.copy()]
                    correct_masks = [*correct_masks, *correct_masks_tmp.copy()]

        return images_path, labels, id_per_sample, masks, correct_masks


    def __len__(self):
        return len(self.images)


    def __getitem__(self, idx):
        img = cv2.imread(self.images[idx])

        if self.use_masks:
            mask = cv2.imread(self.masks[idx], cv2.IMREAD_GRAYSCALE)
            if self.correct_masks[idx] != mask.size:
                img[mask < 50] = (255, 255, 255)

        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        label = torch.tensor((self.labels[idx]), dtype=(torch.long)).item()
        img = self.transform(img)
        return img, label, torch.tensor(idx, dtype=(torch.long)).item()


    def get_template_name(self, idx):
        image_path = self.images[idx][0]
        image_filename = Path(image_path).parent.parent.name
        return image_filename
    
    def get_num_identities(self):
        return len(set(self.labels))


class ResetIndicesDS(Dataset):
    def __init__(self, subset):
        self.subset = subset
        unique_targets = sorted(set(target for _, target in subset))
        self.target_map = {old_target: torch.tensor(new_target, dtype=torch.long) for new_target, old_target in enumerate(unique_targets)}
        print(f'[INFO] Unique targets: {len(unique_targets)}')

    def __getitem__(self, index):
        image, old_target = self.subset[index]
        new_target = self.target_map[old_target]  # remap the target here
        return image, new_target
    def __len__(self):
        return len(self.subset)

