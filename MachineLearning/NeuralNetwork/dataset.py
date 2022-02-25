import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

import os

class Qlsa(Dataset):
    def __init__(self, dataset:str, train:bool, transform=None):
        if train:
            self.img_labels = pd.read_csv(os.path.dirname(__file__) + '\\datasets\\' + dataset + '\\train.csv')
        else:
            self.img_labels = pd.read_csv(os.path.dirname(__file__) + '\\datasets\\' + dataset + '\\test.csv')
        self.img_dir = os.path.dirname(__file__) + '\\datasets\\' + dataset + '\\images'
        self.transform = transform
        self.target_transform = None

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('L')
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

