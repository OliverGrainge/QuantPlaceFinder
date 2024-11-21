import os
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from torch.utils.data import Dataset


class Tokyo247(Dataset):
    def __init__(self, val_dataset_dir=None, input_transform=None):
        self.input_transform = input_transform
        self.dataset_root = os.path.join(val_dataset_dir, "Tokyo247")

        # reference images names
        self.dbImages = np.load("image_paths/tokyo247_dbImages.npy")

        # query images names
        self.qImages = np.load("image_paths/tokyo247_qImages.npy")

        # ground truth
        self.ground_truth = np.load(
            "image_paths/tokyo247_gt.npy", 
            allow_pickle=True
        )

        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))

        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataset_root, self.images[index]))
        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)