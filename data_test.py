import numpy as np
import torch
from model import Fast_Robust_Curve_Net
import torch.nn.functional as F
from PIL import Image

device = 'cuda'

def Test_img(img_test1):
    Netshow = Fast_Robust_Curve_Net(24).to(device)
    Netshow.load_state_dict(torch.load(
        "train_result/Net_weight.pkl"
    )['weight'])

    Netshow.eval()

    img_test1 = img_test1.cuda()

    with torch.no_grad():
        out = Netshow(img_test1)

    return out