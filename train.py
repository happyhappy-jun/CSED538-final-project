import torch
from efficientnet_pytorch import EfficientNet
from torch.utils.data import DataLoader
import pandas as pd
from Augplus import LoadDataset
from model import *
from datetime import datetime

SAVE_STATE_DICT = True # if enabled only save parameter, false: save whole model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name(0))
data_cfgs = {"name": "DL20", "num_classes": 20, "dir": "DL20"}
train_cfgs = {"batch_size": 64, "lr": 0.001, "min_lr": 0.0001, "total_epoch": 20, "model": "efficientnet-B7"}

### load small version of ResNet
# model = Small_ResNet(BasicBlock, [3, 3, 3], num_classes=data_cfgs['num_classes']).to('cuda')
model = EfficientNet.from_name('efficientnet-b0', num_classes=20)
# model = EfficientNet.from_pretrained('efficientnet-b7',  num_classes=20)
# model = timm.create_model('tresnet_xl', pretrained=True, num_classes=data_cfgs["num_classes"])
if torch.cuda.is_available():
    model.cuda()

### load train/valid/test dataset
train_dataset = LoadDataset(data_cfgs["dir"], mode="valid", is_train=True)
valid_dataset = LoadDataset(data_cfgs["dir"], mode="valid", is_train=False)
# test_dataset = LoadDataset(data_cfgs["dir"], mode="test", random_flip=False)

### warp dataset using dataloader
train_dataloader = DataLoader(train_dataset, batch_size=train_cfgs["batch_size"], shuffle=True, pin_memory=True,
                              drop_last=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=train_cfgs["batch_size"], shuffle=False, pin_memory=True,
                              drop_last=False)
# test_dataloader = DataLoader(test_dataset, batch_size=train_cfgs["batch_size"], shuffle=False, pin_memory=True, drop_last=False)

### define Adam optimizer: one of the popular optimizers in Deep Learning community
optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, nesterov=True, lr=train_cfgs["lr"])

## LR Schedule
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = train_cfgs["total_epoch"], eta_min=train_cfgs["min_lr"])

### define cross-entropy loss for classification
criterion = nn.CrossEntropyLoss()

log = pd.DataFrame(columns=['epochs', 'train_loss', 'test_loss', 'test_accuracy'])
####################################################################################################
### Start training ###
print("Start Training")
model.train()
step, epoch, valid_logging = 0, 0, False
train_iter = iter(train_dataloader)
while epoch <= train_cfgs["total_epoch"]:
    model.train()
    optimizer.zero_grad()
    try:
        images, labels = next(train_iter)
        valid_logging = False
        step += 1
    except StopIteration:
        train_iter = iter(train_dataloader)
        images, labels = next(train_iter)
        valid_logging = True
        step += 1
        epoch += 1

    images, labels = images.to(device), labels.to(device)
    logits = model(images)
    loss = criterion(logits, labels)

    loss.backward()
    optimizer.step()
    if step % 100 == 0:
        print("Step: {step} \t Loss: {loss}".format(step=step, loss=loss.item()))

    if valid_logging:
        correct, total = 0, 0
        model.eval()
        with torch.no_grad():
            for images, labels in iter(valid_dataloader):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                test_loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print(f'\nEpoch {epoch} Accuracy of the network on the {len(valid_dataset)} valid images: \
              {100 * correct / total}')
        log.loc[len(log)] = [epoch, loss, test_loss, 100 * correct / total]
    # update learning ratew
    scheduler.step()

log.to_csv(train_cfgs["model"]+"_log.csv", index=False)

if SAVE_STATE_DICT:
    torch.save(model.state_dict(), train_cfgs["model"]+".h5")
else:
    torch.save(model, train_cfgs["model"]+".h5")