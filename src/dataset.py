from functools import lru_cache
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
        is_rgb=False, mode=None
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

        # self.morph_kernel = np.ones([(self.k_size[0]+self.k_size[1])//2*2+1])
        self.morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, [self.k_size[1]]*2)
        
        # self.init_files()

    # @lru_cache(128)
    def read_files(self, idx):
        idx_light = "%02d" % (idx % self.n_lights)
        idx = idx // self.n_lights
        path = self.imgpath_list[idx]
        img_name = self.img_list[idx]
        
        sh_light = torch.from_numpy(np.loadtxt(os.path.join(path, f"{img_name}_light_{idx_light}.txt")))
        img_orig = Image.open(os.path.join(path, f"{img_name}_{idx_light}.png")).copy()
        mask_shadow = Image.open(os.path.join(path, f"{img_name}_shadowmask_{idx_light}.png")).copy()
        mask_face = Image.fromarray(np.load(os.path.join(path, f"{img_name}_faceregion.npy")).astype(np.uint8)*255)
        return img_orig, mask_shadow, mask_face, sh_light

    # def init_files(self):
    #     self.sh_lights = []
    #     self.img_origs = []
    #     self.mask_shadows = []
    #     self.mask_faces = []
    #     idx_lights = ["%02d"%i for i in range(self.n_lights)]
    #     for i in range(len(self.img_list)):
    #         path = self.imgpath_list[i]
    #         img_name = self.img_list[i]
    #         for idx_light in idx_lights:
    #             self.sh_lights.append(torch.from_numpy(np.loadtxt(os.path.join(path, f"{img_name}_light_{idx_light}.txt"))))
    #             self.img_origs.append(Image.open(os.path.join(path, f"{img_name}_{idx_light}.png")).copy())
    #             self.mask_shadows.append(Image.open(os.path.join(path, f"{img_name}_shadowmask_{idx_light}.png")).copy())
    #             self.mask_faces.append(Image.fromarray(np.load(os.path.join(path, f"{img_name}_faceregion.npy")).astype(np.uint8)*255))

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
                T.RandomAffine(5, shear=15),
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
        # sh_light = self.sh_lights[idx]
        # img_orig = self.img_origs[idx]
        # mask_shadow = self.mask_shadows[idx]
        # mask_face = self.mask_faces[idx]
        
        img_orig, mask_shadow, mask_face, sh_light = self.read_files(idx)

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
        
        self.get_shadow_weight((1-mask_shadow)*mask_face) # 1 for shadow

        return self.to_tensor(img_shadow), self.to_tensor(img_orig), self.to_tensor(mask_shadow), sh_light

    def get_shadow_weight(self, mask):
        # mask: (h, w), value = float32 ~ [0, 1]
        return (cv2.dilate((mask*255).astype(np.uint8), self.morph_kernel))/255.

class DPRShadowDataset_ColorJitter(DPRShadowDataset):
    def __init__(self, root, img_list=None, n_lights=5, size=256, intensity=(0.1, 0.7), k_size=(0.02, 0.1), is_rgb=False, mode='mixed'):
        super().__init__(root, img_list=img_list, n_lights=n_lights, size=size, intensity=intensity, k_size=k_size, is_rgb=is_rgb)
        # is_rgb is useless here
        self.cj = torchvision.transforms.ColorJitter(brightness=(0.5, 1.0), contrast=(0.3)) if mode == 'mixed' \
            else torchvision.transforms.ColorJitter(brightness=(0.3, 1.0), contrast=(0.3), saturation=(0.5), hue=0.1)

        self.make_shadow_dict = {
            'mixed': self.make_shadow_mixed,
            'colormatrix': lambda img_orig, mask_face: self.make_shadow_color_matrix(np.array(img_orig)/255., mask_face),
            'torchcj': self.make_shadow_torch_cj,
        }
        assert mode in self.make_shadow_dict, f'mode({mode}) should be one of {list(self.make_shadow_dict.keys())}'
        print("DPRShadowDataset_ColorJitter, mode: " + mode)
        self.make_shadow = self.make_shadow_dict[mode]
        
        # self.gain = ()
        
    
    def make_shadow_color_matrix(self, img_orig, mask_face):
        img_jitter = apply_color_gain(
            img_orig[mask_face].reshape(-1, 3),
            # self.gain)
            np.random.uniform(*self.intensity, size=(1, 3)))
        return np.clip(img_orig @ approx_color_matrix(img_orig[mask_face].reshape(-1, 3), img_jitter), 0, 1)

    # def make_shadow_tonejitter_raw(self, img_orig, mask_face): #TODO: rm
    #     return apply_color_gain(img_orig, self.gain)

    def make_shadow_torch_cj(self, img_orig, mask_face):
        return np.array(self.cj(img_orig))/255.
    
    def make_shadow_mixed(self, img_orig, mask_face):
        # Mix approch from torchvision.transforms.ColorJitter + Tone-mapping ColorJitterMatrix 
        return self.make_shadow_color_matrix(
            self.make_shadow_torch_cj(img_orig, mask_face),
            mask_face
        )

    def __getitem__(self, idx):
        # if np.random.uniform(0, 1) >= 0.5 : 
        #     return super().__getitem__(idx)

        # sh_light = self.sh_lights[idx]
        # img_orig = self.img_origs[idx]
        # mask_shadow = self.mask_shadows[idx]
        # mask_face = self.mask_faces[idx]
        
        img_orig, mask_shadow, mask_face, sh_light = self.read_files(idx)

        img_orig = self.transform(img_orig)
        img_shadow = img_orig.copy()
        img_orig = np.array(img_orig).astype(np.float32)/255 # float32

        mask_face = self.mask_transform(mask_face) > 1 # uint8 -> bool
        mask_shadow = self.shadow_transform(mask_shadow) # float32
        
        # self.gain = np.random.uniform(*self.intensity, size=(1, 3)) # TODO:rm
        # img_shadow2 = self.make_shadow_tonejitter_raw(np.array(img_shadow)/255., mask_face) # TODO: rm
        img_shadow = self.make_shadow(img_shadow, mask_face) # f(PIL.Image) -> float32:[0, 1]

        mask_shadow = ((1-mask_shadow)*mask_face)[..., np.newaxis] # 1 for shadow
        img_shadow = (img_orig*(1-mask_shadow) + img_shadow*mask_shadow).astype(np.float32)
        # img_shadow2 = (img_orig*(1-mask_shadow) + img_shadow2*mask_shadow).astype(np.float32) #TODO: rm
        # t = mask_shadow.copy()

        mask_shadow = mask_shadow.squeeze()
        # mask_shadow = self.get_shadow_weight(mask_shadow.squeeze())

        # mask_shadow = np.clip(mask_shadow*5, 0, 1)
        # return self.to_tensor(img_orig), self.to_tensor(img_shadow2), self.to_tensor(img_shadow), self.to_tensor(mask_shadow), sh_light
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