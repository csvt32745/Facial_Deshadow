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

# Color Jitter Tranform from Portrait Shadow Manipulation (SIGGRAPH 2020)
# https://github.com/google/portrait-shadow-manipulation/blob/master/utils.py

def schlick_bias(x, bias):
    return x / ((1.0 / bias - 2.0) * (1.0 - x) + 1.0 + 1e-6)

def apply_color_gain(image, gain):
    """Apply tone perturbation to images.
    Tone curve jitter comes from Schlick's bias and gain.
    Schlick, Christophe. "Fast alternatives to Perlin’s bias and gain functions." Graphics Gems IV 4 (1994).
    Args:
    image: a 3D image tensor [H, W, C].
    gain: a tuple of length 3 that specifies the strength of the jitter per color channel.
    is_rgb: a bool that indicates whether input is grayscale (C=1) or rgb (C=3).

    Returns:
    3D tensor applied with a tone curve jitter, has the same size as input.
    """
    mask = image >= 0.5
    new_image = (schlick_bias(image * 2.0, gain) / 2.0 * (1.0 - mask)) \
        + ((schlick_bias(image * 2.0 - 1.0, 1.0 - gain) / 2.0 + 0.5) * mask)
    return new_image

def approx_color_matrix(image, new_image):
    return np.linalg.lstsq(image.reshape(-1 ,3), new_image.reshape(-1, 3), rcond=None)[0]