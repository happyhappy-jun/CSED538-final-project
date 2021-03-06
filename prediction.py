import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from dataloader import LoadDataset

data_cfgs = {"name": "DL20", "num_classes": 20, "dir": "DL20"}
train_cfgs = {"batch_size": 32, "lr": 0.0002, "total_epoch": 20, "model_name": "TRESNET_XL_PRETRAINED"}

test_dataset = LoadDataset(data_cfgs["dir"], mode="test", is_train=False)
test_dataloader = DataLoader(test_dataset, batch_size=train_cfgs["batch_size"], shuffle=False, pin_memory=True,
                             drop_last=False)
# Load trained model

# You need to choose correct model type to use load_stat_dict feature
# this boolean value should match with SAVE_STATE_DICT in train.py
LOAD_FROM_STATE_DICT = True
if LOAD_FROM_STATE_DICT:
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=20)
    model.load_state_dict(torch.load(train_cfgs["model_name"] + ".h5"))
else:
    model = torch.load(torch.load(train_cfgs["model_name"] + ".h5"))

model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

predictions = []
for images, labels in iter(test_dataloader):
    images, _ = images.to(device), labels.to(device)
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    predictions += predicted.tolist()

id = [os.path.splitext(f)[0] for f in os.listdir("DL20/test/20") if f.endswith('.png')]
id.sort()

submission = pd.DataFrame({"id": id, "Category": predictions})
submission.to_csv("submission.csv", index=False)
