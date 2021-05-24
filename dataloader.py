
import os

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

from PIL.Image import BICUBIC

class LoadDataset(Dataset):
    def __init__(self, data_path, mode, is_train=False):
        super(LoadDataset, self).__init__()
        self.data_path = data_path
        self.mode = mode
        self.random_flip = is_train
        self.norm_mean =  (0.485, 0.456, 0.406)
        self.norm_std = (0.229, 0.224, 0.225)

        self.transforms = []
        if is_train:
            self.transforms = [transforms.Resize(255, BICUBIC),
                               transforms.RandomCrop(224),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               transforms.Normalize(self.norm_mean, self.norm_std)
                               ]
        else:
            self.transforms = [transforms.Resize(255, BICUBIC),
                               transforms.RandomCrop(224),
                               transforms.ToTensor(),
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
