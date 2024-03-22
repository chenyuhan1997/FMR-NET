from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import pickle
from PIL import Image
import os
import torchvision
from model import Fast_Robust_Curve_Net
from data_test import Test_img
import torchvision
from torchvision import transforms

device='cuda'

test_data_path = '.\\test_data\\'

tfs_full = transforms.Compose([
        transforms.ToTensor()
    ])
Test_Image_Number=len(os.listdir(test_data_path))
print(Test_Image_Number)

for i in range(int(Test_Image_Number)):

    Test_low = Image.open(test_data_path+str(i+1)+'.jpg').convert('RGB')
    low = tfs_full(Test_low).unsqueeze(0).to(device)
    end = Test_img(low)

    torchvision.utils.save_image(end,'kkk'+'%d.jpg' % i, padding = 0)