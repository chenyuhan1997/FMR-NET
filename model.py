import os
import numpy as np
import os.path as osp
import cv2
import argparse
import time
import torch
from torch.utils.data import DataLoader
import torchvision
import torch.nn as nn
import scipy.io as sio
import torch.nn.functional as F
import math
from thop import profile


'''
You can choose to add the following operators to put into the MRB, 
the pre-training model of this code uses only the traditional Multi-scale convolution
'''

# ----------------------------------------------------------------------------------------------------------------------
class laplacian(nn.Module):
    def __init__(self, channels):
        super(laplacian, self).__init__()
        laplacian_filter = np.array([[0, 1, 0],
                                     [1, -4, 1],
                                     [0, 1, 0]])
        self.conv_x = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
                                stride=1, dilation=1, groups=channels, bias=False)
        self.conv_x.weight.data.copy_(torch.from_numpy(laplacian_filter))
        self.conv_y = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
                                stride=1, dilation=1, groups=channels, bias=False)
        self.conv_y.weight.data.copy_(torch.from_numpy(laplacian_filter.T))
    def forward(self, x):
        laplacianx = self.conv_x(x)
        laplaciany = self.conv_y(x)
        x = torch.abs(laplacianx) + torch.abs(laplaciany)
        return x
#-----------------------------------------------------------------------------------------------------------------------
class Sobelxy(nn.Module):
    def __init__(self, channels):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.conv_x = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
                                stride=1, dilation=1, groups=channels, bias=False)
        self.conv_x.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.conv_y = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1,
                                stride=1, dilation=1, groups=channels, bias=False)
        self.conv_y.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.conv_x(x)
        sobely = self.conv_y(x)
        x = torch.abs(sobelx) + torch.abs(sobely)
        return x
#-----------------------------------------------------------------------------------------------------------------------
class MRB(nn.Module):
    def __init__(self, in_feature):
        super(MRB, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=in_feature, out_channels=2 * in_feature, kernel_size=1, stride=1, padding=0, bias=True)
        self.Conv2_1 = nn.Conv2d(in_channels=2 * in_feature, out_channels=2 * in_feature, kernel_size=1, stride=1, padding=0, bias=True)
        self.Conv2_2 = nn.Conv2d(in_channels=2 * in_feature, out_channels=2 * in_feature, kernel_size=3, stride=1, padding=1, bias=True)
        self.Conv2_3 = nn.Conv2d(in_channels=2 * in_feature, out_channels=2 * in_feature, kernel_size=5, stride=1, padding=2, bias=True)
        self.Conv2_4 = nn.Conv2d(in_channels=2 * in_feature, out_channels=2 * in_feature, kernel_size=7, stride=1, padding=3, bias=True)
        self.Conv3 = nn.Conv2d(in_channels=2 * in_feature, out_channels=in_feature, kernel_size=1, stride=1, padding=0, bias=True)
        self.LRelu = nn.LeakyReLU()
    def forward(self, x):
        out1 = self.LRelu(self.Conv1(x))
        out21 = self.LRelu(self.Conv2_1(out1))
        out22 = self.LRelu(self.Conv2_2(out1))
        out23 = self.LRelu(self.Conv2_3(out1))
        out24 = self.LRelu(self.Conv2_4(out1))
        out3 = torch.add(out1, (out21 + out22 + out23 + out24))
        out4 = self.LRelu(self.Conv3(out3))
        out5 = torch.add(x, out4)
        return out5
#-----------------------------------------------------------------------------------------------------------------------
class CA(nn.Module):
    def __init__(self):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)
        return x * out.expand_as(x)
#-----------------------------------------------------------------------------------------------------------------------
class FMRB(nn.Module):
    def __init__(self, in_feature):
        super(FMRB, self).__init__()
        self.MRB = MRB(in_feature)
        self.CA = CA()
        self.LRelu = nn.LeakyReLU()
    def forward(self, x):
        out = self.LRelu(self.MRB(x))
        out = self.MRB(out)
        out = self.CA(out)
        return out
#----------------------------------------- ------------------------------------------------------------------------------
class Fast_Robust_Curve_Net(nn.Module):
    def __init__(self, in_feature):
        super(Fast_Robust_Curve_Net, self).__init__()
        self.Conv_in = nn.Conv2d(in_channels=3, out_channels=in_feature, kernel_size=3, stride=1, padding=1, dilation=1, bias=True)
        self.FMRB = FMRB(in_feature)
        self.Conv_end = nn.Conv2d(in_channels=in_feature, out_channels=3, kernel_size=1, stride=1, padding=0, bias=True)
        self.lRelu = nn.LeakyReLU()
        # self.sol = Sobelxy(in_feature)
        # self.lap = laplacian(in_feature)
    def forward(self, x):
        out_in = self.lRelu(self.Conv_in(x))
        # out_s = self.sol(out_in)
        # out_l = self.lap(out_in)
        out1 = self.FMRB(out_in)
        out2 = self.FMRB(out1)
        out3 = self.FMRB(torch.add(out2, out_in))
        #---------------------------------------------------------------------------------------------------------------
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(out2, 3, dim=1)
        y_1 = x + r1 * (torch.pow(x, 2) - x)
        y_2 = y_1 + r2 * (torch.pow(y_1, 2) - y_1)
        y_3 = y_2 + r3 * (torch.pow(y_2, 2) - y_2)
        y_4 = y_3 + r4 * (torch.pow(y_3, 2) - y_3)
        y_5 = y_4 + r5 * (torch.pow(y_4, 2) - y_4)
        y_6 = y_5 + r6 * (torch.pow(y_5, 2) - y_5)
        y_7 = y_6 + r7 * (torch.pow(y_6, 2) - y_6)
        enhance = y_7 + r8 * (torch.pow(y_7, 2) - y_7)
        enhance = self.lRelu(self.Conv_in(enhance))
        out_e = self.FMRB(torch.add(enhance, out3))
        out = self.lRelu(self.Conv_end(out_e))
        return out

if __name__ == "__main__":
    model = Fast_Robust_Curve_Net(24)
    input = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, inputs=(input, ))
    print("flops:{}".format(flops))
    print("params:{}".format(params))