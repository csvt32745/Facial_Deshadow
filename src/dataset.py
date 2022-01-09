import types
import os
import numpy as np
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
import torchvision
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader
from src.utils import fuse_shadow
from PIL import Image

from src.utils import denormalize_img_rgb, RGB2LAB, apply_color_gain, approx_color_matrix

class DPRShadowDataset(Dataset):
    def __init__(self, root, img_list=None, 
        n_lights=5, size=256, 
        intensity=(0.1, 0.7), k_size=(0.02, 0.1),
        is_rgb=False
        ):

        # root/imgHQxxxxx/
        base_img_list = os.listdir(root) if img_list is None else img_list
        self.imgpath_list = list(filter(os.path.isdir, [os.path.join(root, p) for p in base_img_list]))
        self.img_list = [os.path.split(p)[1] for p in self.imgpath_list]
        self.n_lights = n_lights
        self.k_size = [int(size*k) for k in k_size] # odd-size kernel size of blur on shadow
        self.k_size[1] = max(*self.k_size)
        
        self.intensity = list(intensity) # shadow intensity
        self.intensity[1] = max(self.intensity)
        self.size = size
        self.init_transform(size)

        self.is_rgb = is_rgb

    def init_transform(self, size):
        self.transform = T.Compose(
            [
                T.Resize((size, size)),
                # T.Lambda(RGB2LAB),
                # T.Lambda(lambda img: np.array(img)/255)
            ]
        )
        self.shadow_transform = T.Compose(
            [
                T.Resize((size, size)),
                T.Lambda(
                    lambda img: T.functional.gaussian_blur(img, 1+2*np.random.randint(*self.k_size), sigma=None)
                ) if self.k_size[0] != self.k_size[1] else T.Lambda(
                    lambda img: T.functional.gaussian_blur(img, 1+2*self.k_size[0], sigma=None)
                ),
                T.Lambda(lambda img: np.array(img)/255)
            ]
        )

        self.mask_transform = T.Compose(
            [
                T.Resize((size, size)),
                T.Lambda(lambda img: np.array(img))
            ]
        )

        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.img_list)*self.n_lights

    def __getitem__(self, idx):
        idx_light = "%02d" % (idx % self.n_lights)
        idx = idx // self.n_lights
        path = self.imgpath_list[idx]
        img_name = self.img_list[idx]

        sh_light = torch.from_numpy(np.loadtxt(os.path.join(path, f"{img_name}_light_{idx_light}.txt")))
        img_orig = Image.open(os.path.join(path, f"{img_name}_{idx_light}.png"))
        mask_shadow = Image.open(os.path.join(path, f"{img_name}_shadowmask_{idx_light}.png"))
        mask_face = Image.fromarray(np.load(os.path.join(path, f"{img_name}_faceregion.npy")).astype(np.uint8)*255)
        
        img_orig = self.transform(img_orig) # uint8
        img_orig_lab = RGB2LAB(img_orig)
        mask_face = self.mask_transform(mask_face) > 1 # uint8 -> bool
        mask_shadow = self.shadow_transform(mask_shadow) # float32
        
        img_shadow = fuse_shadow(
            img_orig_lab[..., 0]/255.,
            mask_face,
            mask_shadow,
            np.random.uniform(*self.intensity)).astype(np.float32)
        
        if self.is_rgb:
            img_shadow = cv2.cvtColor(
                np.stack([
                    (np.clip(img_shadow, 0, 1)*255).astype(np.uint8),
                    img_orig_lab[..., 1], img_orig_lab[..., 2]], axis=-1),
                cv2.COLOR_Lab2RGB
            ) # 3-ch RGB
        else:
            img_orig = img_orig_lab

        return self.to_tensor(img_shadow), self.to_tensor(img_orig), sh_light

class DPRShadowDataset_ColorJitter(DPRShadowDataset):
    def __init__(self, root, img_list=None, n_lights=5, size=256, intensity=(0.1, 0.7), k_size=(0.02, 0.1), is_rgb=False):
        super().__init__(root, img_list=img_list, n_lights=n_lights, size=size, intensity=intensity, k_size=k_size, is_rgb=is_rgb)
        # is_rgb is useless here
        # self.cj = torchvision.transforms.ColorJitter(brightness=(0.3, 1.0), contrast=(0.3), saturation=(0.5), hue=0.1)
        self.cj = torchvision.transforms.ColorJitter(brightness=(0.5, 1.0), contrast=(0.3))
        # self.morph_kernel = np.ones([(self.k_size[0]+self.k_size[1])//2*2+1])
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [self.k_size[1]]*2)

    def __getitem__(self, idx):
        idx_light = "%02d" % (idx % self.n_lights)
        idx = idx // self.n_lights
        path = self.imgpath_list[idx]
        img_name = self.img_list[idx]

        sh_light = torch.from_numpy(np.loadtxt(os.path.join(path, f"{img_name}_light_{idx_light}.txt")))
        img_orig = Image.open(os.path.join(path, f"{img_name}_{idx_light}.png"))
        mask_shadow = Image.open(os.path.join(path, f"{img_name}_shadowmask_{idx_light}.png"))
        mask_face = Image.fromarray(np.load(os.path.join(path, f"{img_name}_faceregion.npy")).astype(np.uint8)*255)
        
        img_orig = self.transform(img_orig)
        img_jitter = img_orig.copy()
        img_orig = np.array(img_orig).astype(np.float32)/255 # float32

        mask_face = self.mask_transform(mask_face) > 1 # uint8 -> bool
        mask_shadow = self.shadow_transform(mask_shadow) # float32
        # Mix approch from torchvision.transforms.ColorJitter + Tone-mapping ColorJitterMatrix 
        img_jitter = np.array(self.cj(img_jitter))/255.
        img_jitter = apply_color_gain(
            img_jitter[mask_face].reshape(-1, 3),
            np.random.uniform(*self.intensity, size=(1, 3)))
        img_jitter = np.clip(img_orig @ approx_color_matrix(img_orig[mask_face].reshape(-1, 3), img_jitter), 0, 1)

        # Data Augmentation from torchivision
        # img_jitter = np.array(self.cj(img_jitter))/255.

        mask_shadow = ((1-mask_shadow)*mask_face)[..., np.newaxis] # 1 for shadow
        img_shadow = (img_orig*(1-mask_shadow) + img_jitter*mask_shadow).astype(np.float32)
        # t = mask_shadow.copy()
        mask_shadow = (cv2.dilate((mask_shadow.squeeze()*255).astype(np.uint8), self.morph_kernel))/255.
        # mask_shadow = np.clip(mask_shadow*5, 0, 1)
        return self.to_tensor(img_shadow), self.to_tensor(img_orig), self.to_tensor(mask_shadow), sh_light


class UnsupervisedDataset(Dataset):
    def __init__(self, root, img_list=None, size=256, is_rgb=False):
        # root/imgHQxxxxx/
        self.img_list = os.listdir(root) if img_list is None else img_list
        self.img_name = list(filter(lambda p: os.path.splitext(p)[1] in ['.jpg', '.png', '.jpeg'], self.img_list))
        self.img_list = [os.path.join(root, p) for p in self.img_name]
        self.is_rgb = is_rgb
        self.init_transform(size)

    def __len__(self):
        return len(self.img_list)

    def init_transform(self, size):
        if self.is_rgb:
            self.transform = T.Compose(
                    [
                        T.Resize((size, size)),
                        T.ToTensor(),
                        # T.Normalize(mean=[0.485, 0.456, 0.406],
                        #         std=[0.229, 0.224, 0.225])
                    ]
                )
        else:
            self.transform = T.Compose(
                    [
                        T.Resize((size, size)),
                        T.Lambda(RGB2LAB),
                        T.ToTensor(),
                    ]
                )

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx])
        img = self.transform(img)
        return img

if __name__ == '__main__':
    d = DPRShadowDataset('../DPR_dataset')
    print(len(d))
    print(d[len(d)//2])