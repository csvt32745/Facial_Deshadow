import os
import sys
sys.path.append('sh_lighting')
sys.path.append('RI_render_DPR/step_4')
sys.path.append('RI_render_DPR/step_6')
sys.path.append('RI_render_DPR/utils')

import numpy as np
import trimesh
import cv2
import mediapy as media
import itertools
from tqdm import tqdm

# external script
from sh_lighting import *
from get_pixelValue import *
from utils_normal import *

from ibug.face_detection import RetinaFacePredictor as FacePredictor
from ibug.face_parsing.utils import label_colormap
from ibug.face_parsing import FaceParser

def ray_intersect(flat_vertices, light_dir, ray_bias=0.07):
    """
    flat_vertices: (N, 3) for 3d points on the image
    light_dir: 3d light direction
    return (N,) bool array
    """
    light_dir = np.tile(light_dir, [flat_vertices.shape[0], 1])
    flat_vertices = flat_vertices + light_dir * ray_bias
    result = mesh.ray.intersects_any(flat_vertices, light_dir)
    return result

def find_main_light_direction(sh_light, n_sample=120, top_avg=0.1):
    """
    sh_light: 9 coeff as (9, 1)
    """
    phi, theta = np.random.default_rng().uniform(0, 1, (2, n_sample))
    phi, theta = 2*np.pi*phi, np.arccos(2*theta-1)
    rd_dirs = np.array([[np.sin(t)*np.cos(p), np.sin(t)*np.sin(p), np.cos(t)] for p, t in zip(phi, theta)])
    sh_dirs = SHBasis(rd_dirs)
    best_dir = rd_dirs[np.argsort((sh_dirs @ sh_light).flatten())[-int(n_sample*top_avg):]].mean(axis=0).reshape(1, 3)
    return best_dir

def get_warped_soft_shadow(shape, img_shadow, warp_uv, filter_size):
    shadow = textureSampling(img_shadow.reshape(*shape, 1), warp_uv).astype(np.float32)
    # warp_shadow = cv2.boxFilter(shadow.astype(float), -1, (51, 51))
    warp_shadow = cv2.blur(shadow, (filter_size, filter_size))
    # warp_shadow = cv2.GaussianBlur(shadow.astype(float), (51, 51), 101)
    return shadow, warp_shadow

def fuse_shadow(img_orig, mask, shadow, intensity=0.4):
    """ 
    img_orig: original image
    mask: human mask of the image
    shadow: array_like(mask) in [0, 1], 1 for shadowed(ray hit)
    intensity: weight for shadow
    """
    lab = cv2.cvtColor(img_orig.astype(np.float32), cv2.COLOR_RGB2Lab)

    lum = lab[:, :, 0].reshape(-1)
    lum[mask] = lum[mask] * (1 - (1-shadow[mask].reshape(-1)) * intensity)
    lab[:, :, 0] = lum.reshape(lab.shape[:2])

    img_res = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Mask vis
    # img_vis = img_orig.copy().reshape(-1, 3)
    # img_vis[mask] *= np.array([[2, 1, 1]])
    # img_vis = img_vis.reshape(h, w, 3)
    return img_res


if __name__ == '__main__':
    
    face_list = []
    with open('RI_render_DPR/data.list') as f:
        face_list += f.readlines()
    face_list = sorted(face_list)

    result_root = 'RI_render_DPR/result'
    # relight_root = 'RI_render_DPR/relighting'
    relight_root = 'DPR_dataset'

    face_predictor = FacePredictor(model=(FacePredictor.get_model('mobilenet0.25')))
    face_parser = FaceParser(num_classes=14)

    tqdm_face_list = tqdm(face_list, ascii=True, dynamic_ncols=True)
    defected_face = []
    for file_name in tqdm_face_list:
        # imgHQ00000_00.png
        img_name = file_name.split(".")[0] 
        # dataset_img_name = file_name.split(".")[0] 
        dataset_img_name = file_name.split("_")[0] # Since imgHQ00000_00.png
        tqdm_face_list.set_description_str(desc=img_name)
        path = os.path.join(result_root, img_name)
        relight_path = os.path.join(relight_root, dataset_img_name)

        if os.path.isfile(os.path.join(relight_path, dataset_img_name + f"_shadowmask_04.png")):
            continue

        """
        Processing before warping
        """

        mesh = trimesh.load_mesh(os.path.join(path, img_name+'_new.obj'))
        mask = cv2.imread(os.path.join(path, 'render/mask.png'), 0).astype(bool).reshape(-1)
        img_vertices = np.load(os.path.join(path, 'render/vertex_3D.npy'))
        # normal = np.load(os.path.join(path, 'warp/full_normal_faceRegion_faceBoundary_extend.npy'))
        shape = img_vertices.shape[:2]
        img_vertices = img_vertices.reshape(-1, 3)
        
        warp_uv = np.load(os.path.join(path, 'warp/UV_warp.npy')).reshape(-1, 2)
        # warp_mask = cv2.imread(os.path.join(path, 'warp/mask.png'), 0).astype(bool).reshape(-1)
    

        # Label map for 14 classes (or 11 classes): 
        # ```
        # 0 : background
        # 8 : inner_mouth
        # 10 : hair
        # 13 : glasses
        # ```

        parser_input = cv2.imread(os.path.join(relight_path, dataset_img_name + f"_00.png"))
        face = face_predictor(parser_input, rgb=False)

        if len(face) > 1:
            print(f"{img_name} has multiple({len(face)}) face detection!!")
            defected_face.append(img_name)

        elif len(face) == 0:
            print(f"{img_name} cannot detect face!!")
            defected_face.append(img_name)
            continue

        label = face_parser.predict_img(parser_input, face, rgb=False)
        seg_mask = ~np.isin(label, [0, 10, 8, 13])
        seg_mask = seg_mask[0].flatten()

        """ Per light processing """
        for light_id in [f'{i:02d}' for i in range(5)]:
            
            sh_light = np.loadtxt(os.path.join(relight_path, dataset_img_name + f"_light_{light_id}.txt")).reshape(-1, 1)
            img_orig = (cv2.imread(os.path.join(relight_path, dataset_img_name + f"_{light_id}.png"))/255.)[..., ::-1]

            light_dir = find_main_light_direction(sh_light, n_sample=120)
            is_hit = ray_intersect(img_vertices[mask], light_dir[0], ray_bias=0.05)

            img_shadow = np.zeros_like(mask, dtype=float)
            img_shadow[mask] = 1 - is_hit
            shadow_mask, warp_shadow = get_warped_soft_shadow(shape, img_shadow, warp_uv, filter_size=51)

            img_res = fuse_shadow(img_orig, seg_mask, warp_shadow)[..., ::-1]
            cv2.imwrite(os.path.join(relight_path, dataset_img_name + f"_shadow_{light_id}.png"),
                img_res*255)
            cv2.imwrite(os.path.join(relight_path, dataset_img_name + f"_shadowmask_{light_id}.png"),
                shadow_mask.reshape(shape)*255)
            # raise

    if defected_face:
        with open('defected_shadow.list', 'w') as f:
            f.writelines([i+"\n" for i in defected_face])

    with open(os.path.join(relight_root, 'data.list'), 'w') as f:
        f.writelines([p + "\n" for p in face_list])