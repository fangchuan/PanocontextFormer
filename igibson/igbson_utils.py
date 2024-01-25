import json
import os
import numpy as np
import open3d as o3d
from torchvision import transforms
from glob import glob
from torch.utils.data import Dataset

def fibonacci_spiral_samples_on_unit_sphere(nb_samples, mode=0):
    shift = 1.0 if mode == 0 else nb_samples * np.random.random()

    ga = np.pi * (3.0 - np.sqrt(5.0))
    offset = 2.0 / nb_samples

    ss = np.zeros((nb_samples, 3))
    j = 0
    for i in range(nb_samples):
        phi = ga * ((i + shift) % nb_samples)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        cos_theta = ((i + 0.5) * offset) - 1.0
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        ss[j, :] = np.array([sin_phi * sin_theta, cos_theta, cos_phi * sin_theta])
        j += 1
    return ss

def read_json_list(list_file):
    rgb_depth_list = []
    with open(list_file) as f:
        lines = json.load(f)
        for line in lines:
            rgb_name = os.path.join(os.path.split(line)[0],"rgb.png")
            depth_name = os.path.join(os.path.split(line)[0],"depth.png")
            rgb_depth_list.append([rgb_name,depth_name])
    return rgb_depth_list

def read_pkl_list(list_file):
    pkl_list = []
    with open(list_file) as f:
        lines = json.load(f)
        for line in lines:
            pkl_name = os.path.join(os.path.split(line)[0],"data.pkl")
            pkl_list.append(pkl_name)
    return pkl_list

def save_as_point_cloud(depth, rgb, path, mask=None):
    h, w = depth.shape
    Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
    Theta = np.repeat(Theta, w, axis=1)
    Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
    Phi = -np.repeat(Phi, h, axis=0)

    X = depth * np.sin(Theta) * np.sin(Phi)
    Y = depth * np.cos(Theta)
    Z = depth * np.sin(Theta) * np.cos(Phi)

    if mask is None:
        X = X.flatten()
        Y = Y.flatten()
        Z = Z.flatten()
        R = rgb[:, :, 0].flatten()
        G = rgb[:, :, 1].flatten()
        B = rgb[:, :, 2].flatten()
    else:
        X = X[mask]
        Y = Y[mask]
        Z = Z[mask]
        R = rgb[:, :, 0][mask]
        G = rgb[:, :, 1][mask]
        B = rgb[:, :, 2][mask]

    XYZ = np.stack([X, Z, Y], axis=1)

    if(True):
        np.save(path, XYZ)
    else:
        RGB = np.stack([R, G, B], axis=1)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(XYZ)
        pcd.colors = o3d.utility.Vector3dVector(RGB)
        o3d.io.write_point_cloud(path, pcd)

def unitxyz2uv(xyz, equ_w, equ_h):
    x, y, z = np.split(xyz, 3, axis=-1)
    lon = np.arctan2(x, z)
    c = np.sqrt(x ** 2 + z ** 2)
    lat = np.arctan2(y, c)

    # longitude and latitude to equirectangular coordinate
    u = (lon / (2 * np.pi) + 0.5) * equ_w - 0.5
    v = (-lat / np.pi + 0.5) * equ_h - 0.5
    return [u, v]


class Pano3DDataset(Dataset):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    patch_width = 256
    crop_width = 280
    crop_transforms = {}

    def __init__(self, config, mode=None):
        assert isinstance(config, dict)
        self.config = config

        # if save intermedia results as dataset
        save_path = config.get('log', {}).get('save_as_dataset')
        if save_path:
            mode = None
            # copy json split file to dst folder
            os.makedirs(save_path, exist_ok=True)
            split_files = glob(os.path.join(config['data']['split'], '*.json'))
            for split_file in split_files:
                shutil.copy(split_file, os.path.join(save_path, os.path.basename(split_file)))

        # merge train and test set if mode set to None
        self.mode = mode
        if mode == 'val':
            mode = ['test']
        elif mode is None:
            mode = ['train', 'test']
        elif mode in ['train', 'test']:
            mode = [mode]
        else:
            raise Exception("'mode' must be one of 'train', 'test' and None!")

        # load split from json file
        split = config['data']['split']
        if split.endswith('.json'):
            self.root = os.path.dirname(split)
            split_files = [split]
            print(f"Using specified split file {split} with {mode} mode!")
        else:
            self.root = split
            split_files = [os.path.join(self.root, m + '.json') for m in mode]
        self.split = []
        for split_file in split_files:
            if os.path.exists(split_file):
                self.split.extend(read_json(split_file))
            else:
                split = []
                for f in glob(os.path.join(self.root, '*')):
                    if f.lower().endswith(('png', 'jpg')):
                        split.append(os.path.basename(f))
                self.split = split
                break
        self.split = [os.path.join(self.root, folder) for folder in self.split]
        print(f"Dataset mode: {mode}, cameras: {len(self.split)}")

        # crop image argumentation
        self.crop_transforms['train'] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.crop_width, self.crop_width)),
            transforms.RandomCrop((self.patch_width, self.patch_width)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.crop_transforms['test'] = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.patch_width, self.patch_width)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.crop_transforms['val'] = self.crop_transforms['test']

    def __len__(self):
        return len(self.split)

# from igbson.deeppanocontext_utils.igibson_utils import IGScene
# class IGSceneDataset(Pano3DDataset):
#
#     _basic_info = ('name', 'scene', 'camera', 'image_path')
#     def __getitem__(self, index):
#         est_scene, gt_scene = self.get_igibson_scene(index, ('est', 'gt'))
#     def get_igibson_scene(self, item, stype: (str, tuple, list)='gt'):
#         pkl = self.split[item]
#         if pkl.lower().endswith(('png', 'jpg')):
#             gt_scene = IGScene.from_image(pkl)
#             est_scene = None
#         else:
#             if isinstance(stype, str):
#                 stype = (stype, )
#
#             gt_pkl = os.path.join(os.path.dirname(pkl), 'gt.pkl')
#             if os.path.exists(gt_pkl):
#                 est_scene = IGScene.from_pickle(pkl) if 'est' in stype else None
#                 gt_scene = IGScene.from_pickle(gt_pkl, self.igibson_obj_dataset) if 'gt' in stype else None
#             else:
#                 est_scene = None
#                 gt_scene = IGScene.from_pickle(pkl, self.igibson_obj_dataset) if 'gt' in stype else None
#         scenes = {'est': est_scene, 'gt': gt_scene}
#         if len(stype) == 1:
#             return scenes[stype[0]]
#         else:
#             return [scenes[k] for k in stype]
