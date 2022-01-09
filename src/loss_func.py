from os import stat
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from types import MethodType
import cv2
import lpips
from einops import rearrange
from pytorch_msssim import SSIM, MS_SSIM

class GeneratorLoss(nn.Module):
    def __init__(self, is_rgb=False, is_msssim=True):
        super().__init__()
        # self.recon = nn.MSELoss()
        self.recon = nn.L1Loss(reduction='none')
        self.edge = SobelLoss(self.recon, is_rgb)
        self.lpips = lpips.LPIPS(net='vgg')
        # self.ssim = 
        self.ssim = MS_SSIM(data_range=1., size_average=True, channel=3 if is_rgb else 1) if is_msssim \
            else SSIM(data_range=1., size_average=True, channel=3 if is_rgb else 1)
        self.adv = NonSaturaingLoss()

        self.eval()
        
    def forward(self, x, GT, prob, weight=None, ret_dict=True):
        adv = self.adv(prob)
        loss, loss_dict = self.compute_all_woadv(x, GT, weight=weight, ret_dict=ret_dict)
        loss = loss + adv*0.1

        if ret_dict:
            loss_dict["Adv"] = adv.item()
        
        return loss, loss_dict
    
    def compute_all_woadv(self, x, GT, weight=None, ret_dict=True):
        # weight appliable
        if weight is None:
            recon = self.recon(x, GT).mean()
            edge = self.edge(x, GT).mean()
        else:
            recon = (self.recon(x, GT) * weight).mean()
            edge = (self.edge(x, GT) * weight).mean()
        
        perceptual = self.lpips(x, GT, normalize=True).mean()
        ssim = self.ssim(x, GT)

        loss = recon + perceptual + edge + (1-ssim)

        if ret_dict:
            loss_dict = {
                "Recon": recon.item(),
                "LPIPS": perceptual.item(),
                "Edge": edge.item(),
                "SSIM": ssim.item()
            }
            return loss, loss_dict
        return loss, None

class NonSaturaingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        # return (-torch.log(torch.sigmoid(x))).mean(dim=[1, 2, 3]).mean()
        # TEST
        return self.loss_fn(x, torch.ones_like(x, device='cuda'))

class SobelLoss(nn.Module):
    def __init__(self, loss_fn, is_rgb=False):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        sobel = torch.FloatTensor([
            [-1, -1, -1], 
            [-1, 8, -1],
            [-1, -1, -1]
        ]).reshape(1, 1, 3, 3)
        self.conv.weight.data = sobel
        if is_rgb:
            self.gray = nn.Conv2d(3, 1, 1, bias=False)
            self.gray.weight.data = torch.FloatTensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)
            self.conv = nn.Sequential(
                self.gray,
                self.conv
            )
        
        self.loss_fn = loss_fn

    def forward(self, x, y):
        x = self.conv(x)
        y = self.conv(y)
        return self.loss_fn(x, y)
