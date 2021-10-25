import numpy as np
import cv2
import torch


def denormalize_img_rgb(image, to_255=False):
    # shape: np.ndarray | torch.Tensor (..., 3), [-1, 1]
    dim = len(image.shape)
    mean = [0.486, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if isinstance(image, np.ndarray):
        mean = np.array(mean)
        std = np.array(std)
    else:
        mean = torch.FloatTensor(mean)
        std = torch.FloatTensor(std)
    shape = (*([1]*(dim-1)), 3) # len((1, 1, ..., 3)) = dim
    mean = mean.reshape(shape)
    std = std.reshape(shape)
    ret = image*std + mean
    if to_255:
        ret = (np.clip(ret, 0, 1)*255).astype(np.uint8)
    return ret


def RGB2LAB(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2Lab)

def RGB255_TorchLab(img):
    return torch.from_numpy(cv2.cvtColor(img/255., cv2.COLOR_RGB2Lab))

def TorchLab_RGB255(img):
    return cv2.cvtColor((img*255).detach().cpu().numpy().astype(np.uint8), cv2.COLOR_Lab2RGB)