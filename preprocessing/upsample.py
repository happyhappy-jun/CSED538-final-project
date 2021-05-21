import torch
import os
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

trf = transforms.Compose([transforms.ToTensor()])
upsample = torch.nn.Upsample(scale_factor=4)
for (path, dir, files) in os.walk(os.getcwd()+'/upsample_DL20_34k'):
    for filename in files:
        print("%s/%s" % (path, filename))
        image = trf(Image.open(path+"/"+filename))
        images_upsample = upsample(torch.reshape(image, (1, 3, 64, 64)))
        img1 = images_upsample[0]
        save_image(img1, path+"/"+filename)