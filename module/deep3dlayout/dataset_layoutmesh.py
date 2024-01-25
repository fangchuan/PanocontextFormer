import numpy as np
import os
from PIL import Image
from shapely.geometry import LineString
from scipy.spatial.distance import cdist

import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj

MEAN_COLOR_RGB = np.array([0.485, 0.456, 0.406])
MEAN_COLOR_STD = np.array([0.229, 0.224, 0.225])

class PanoLayoutMeshDataset(data.Dataset):
    '''
    See README.md for how to prepare the dataset.
    '''

    def __init__(self, root_dir, split = 'train'):

        if split == "train":
            self.scan_names = self.read_pkl_list(os.path.join(root_dir,'train.json'))
        elif split == "test":
            self.scan_names = self.read_pkl_list(os.path.join(root_dir, 'test.json'))
        elif split == "val":
            self.scan_names = self.read_pkl_list(os.path.join(root_dir, 'test.json'))
        else:
            exit()

        self.img_dir = os.path.join(root_dir, 'igbison_image')
        self.mesh_dir = os.path.join(root_dir, 'layout_mesh')

        self.mesh_fnames =['%s.obj' % fname for fname in self.scan_names]
        self.img_fnames =['%s.png' % fname for fname in self.scan_names]

    def __len__(self):
        return len(self.img_fnames)

    def read_pkl_list(self, list_file):
        import json
        pkl_list = []
        with open(list_file) as f:
            lines = json.load(f)
            for line in lines:
                pkl_name = line.split("/")
                pkl_list.append(pkl_name[0]+"_"+pkl_name[1])
        return pkl_list

    def __getitem__(self, idx):
        # Read image
        img_path = os.path.join(self.img_dir,
                                self.img_fnames[idx])
        img = np.array(Image.open(img_path), np.float32)[..., :3] / 255.
        H, W = img.shape[:2]

        # Prepare image tensor
        img = (img.copy()- MEAN_COLOR_RGB) / MEAN_COLOR_STD

        # Convert all data to tensor
        img_tensor = torch.FloatTensor(img.transpose([2, 0, 1]).copy())

        mesh_filepath = os.path.join(self.mesh_dir,self.mesh_fnames[idx])
        verts, faces, aux = load_obj(mesh_filepath)
        faces_verts_idx = faces.verts_idx


        out_lst = {'image': img_tensor,
                   'gt_mesh_vertics': verts,
                   'gt_mesh_faces': faces_verts_idx,
                   'scan_name':self.img_fnames[idx][:-4]}

        return out_lst

if __name__ == '__main__':
    root_dir = '/Users/yuandong/Documents/Git_project_DAMO/gp3d_private/igibson/example_data'
    layout_dataset = PanoLayoutMeshDataset(root_dir=root_dir)
    train_loader = torch.utils.data.DataLoader(layout_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True,
                                              drop_last=False)
    for batch_idx, batch_data_label in enumerate(train_loader):

        print()
