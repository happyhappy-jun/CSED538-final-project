
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder



class LoadDataset(Dataset):
    def __init__(self, data_path, mode, random_flip=False):
        super(LoadDataset, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.random_flip = random_flip
        self.norm_mean = [0.5,0.5,0.5]
        self.norm_std = [0.5,0.5,0.5]

        self.transforms = []
        if random_flip:
            self.transforms += [transforms.RandomHorizontalFlip()]
        self.transforms += [transforms.ToTensor(),
                            transforms.Normalize(self.norm_mean, self.norm_std)]
        self.transforms = transforms.Compose(self.transforms)

        self.load_dataset()


    def load_dataset(self):
        root = os.path.join(self.data_path, self.mode)
        self.data = ImageFolder(root=root)


    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        img, label = self.data[index]
        img, label = self.transforms(img), int(label)
        return img, label