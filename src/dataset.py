import os
import numpy as np
import cv2
import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms as T
from torch.utils.data.dataloader import DataLoader
from PIL import Image

from src.utils import denormalize_img_rgb, RGB2LAB

default_transform = T.Compose(
    [
        T.Resize(256),
        T.Lambda(RGB2LAB),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406],
        #         std=[0.229, 0.224, 0.225])
    ]
)


class DPRShadowDataset(Dataset):
    def __init__(self, root, img_list=None, n_lights=5, transform=default_transform):
        # root/imgHQxxxxx/
        base_img_list = os.listdir(root) if img_list is None else img_list
        self.imgpath_list = list(filter(os.path.isdir, [os.path.join(root, p) for p in base_img_list]))
        self.img_list = [os.path.split(p)[1] for p in self.imgpath_list]
        
        self.transform = transform
        self.n_lights = n_lights

    def __len__(self):
        return len(self.img_list)*self.n_lights

    def __getitem__(self, idx):
        idx_light = "%02d" % (idx % self.n_lights)
        idx = idx // self.n_lights
        path = self.imgpath_list[idx]
        img_name = self.img_list[idx]

        sh_light = torch.from_numpy(np.loadtxt(os.path.join(path, f"{img_name}_light_{idx_light}.txt")))
        img_orig = Image.open(os.path.join(path, f"{img_name}_{idx_light}.png"))
        img_shadow = Image.open(os.path.join(path, f"{img_name}_shadow_{idx_light}.png"))
        
        img_shadow = self.transform(img_shadow)
        img_orig = self.transform(img_orig)
        return img_shadow, img_orig, sh_light
    
        

# TODO: data augmentation for shadow img producing

if __name__ == '__main__':
    d = DPRShadowDataset('../DPR_dataset')
    print(len(d))
    print(d[len(d)//2])