import torch
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader

from dataloader import LoadDataset
from model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_cfgs = {"name": "DL20", "num_classes": 20, "dir": "DL20"}
train_cfgs = {"batch_size": 50, "lr": 0.001, "min_lr": 0.0001, "total_epoch": 100, "model_name": "Efficient-B7"}

### load small version of ResNet
# model = Small_ResNet(BasicBlock, [3, 3, 3], num_classes=data_cfgs['num_classes']).to('cuda')

model = EfficientNet.from_pretrained('efficientnet-b0',  num_classes=20)
# model = timm.create_model('tresnet_xl', pretrained=True, num_classes=data_cfgs["num_classes"])
if torch.cuda.is_available():
    model.cuda()

### load train/valid/test dataset
train_dataset = LoadDataset(data_cfgs["dir"], mode="train", random_flip=True)
valid_dataset = LoadDataset(data_cfgs["dir"], mode="valid", random_flip=False)
# test_dataset = LoadDataset(data_cfgs["dir"], mode="test", random_flip=False)

### warp dataset using dataloader
train_dataloader = DataLoader(train_dataset, batch_size=train_cfgs["batch_size"], shuffle=True, pin_memory=True,
                              drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=train_cfgs["batch_size"], shuffle=False, pin_memory=True,
                              drop_last=False)
# test_dataloader = DataLoader(test_dataset, batch_size=train_cfgs["batch_size"], shuffle=False, pin_memory=True, drop_last=False)


#Image data Mean & std
from tqdm import tqdm

def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


mean, std = get_mean_std(train_dataloader)
print(mean)
print(std)

mean_val, std_val = get_mean_std(valid_dataloader)
print(mean_val)
print(std_val)