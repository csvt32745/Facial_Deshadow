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

def fuse_shadow_RGB(img_orig, mask, shadow, intensity=0.4):
    """ 
    img_orig: original image
    mask: human mask of the image
    shadow: array_like(mask) in [0, 1], 1 for lit
    intensity: weight for shadow
    """
    lab = cv2.cvtColor(img_orig.astype(np.float32), cv2.COLOR_RGB2Lab)

    lum = lab[:, :, 0]
    lum = lum - (lum * (1-shadow) * intensity) * mask
    # lab[:, :, 0] = lum.reshape(lab.shape[:2])
    img_res = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img_res

def fuse_shadow(lum, mask, shadow, intensity=0.4):
    """ 
    img_orig: original image
    mask: human mask of the image
    shadow: array_like(mask) in [0, 1], 1 for lit
    intensity: weight for shadow
    """
    return lum - (lum * (1-shadow) * intensity) * mask