# coding: utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Dataset for 3D object detection on SUN RGB-D (with support of vote supervision).

A sunrgbd oriented bounding box is parameterized by (cx,cy,cz), (l,w,h) -- (dx,dy,dz) in upright depth coord
(Z is up, Y is forward, X is right ward), heading angle (from +X rotating to -Y) and semantic class

Point clouds are in **upright_depth coordinate (X right, Y forward, Z upward)**
Return heading class, heading residual, size class and size residual for 3D bounding boxes.
Oriented bounding box is parameterized by (cx,cy,cz), (l,w,h), heading_angle and semantic class label.
(cx,cy,cz) is in upright depth coordinate
(l,h,w) are *half length* of the object sizes
The heading angle is a rotation rad from +X rotating towards -Y. (+X is 0, -Y is pi/2)

Author: Charles R. Qi
Date: 2019

"""
import os
import sys
import numpy as np
from torch.utils.data import Dataset
import torch
import scipy.io as sio  # to load .mat files for depth points
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import torch.distributions as dist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(BASE_DIR))

import utils.pc_util as pc_util
import sunrgbd.sunrgbd_utils as sunrgbd_utils
from igibson.model_util_igbson import IGbsonDatasetConfig
from module.layout.dataloader.get_layout_data import LayoutDataset
import trimesh
from shapely.geometry import Polygon
from igibson.dataset_utils import np_coor2xy, np_coory2v
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.structures import Meshes
from module.deep3dlayout.custom_losses.losses import sample_points_from_edges
from igibson.generate_render_trainingdata import generateMeshRenderImage,getTopViewRenderParms
from pytorch3d.io import save_obj,load_obj
from pytorch3d.utils import ico_sphere
import trimesh
from module.depth_estimation.util import Equirec2Cube

DC = IGbsonDatasetConfig()  # dataset specific config
MAX_NUM_OBJ = 64  # maximum number of objects allowed per scene
# MEAN_COLOR_RGB = np.array([0.5, 0.5, 0.5])  # sunrgbd color is in 0~1

MEAN_COLOR_RGB = np.array([0.485, 0.456, 0.406])
MEAN_COLOR_STD = np.array([0.229, 0.224, 0.225])

class IGbsonDetectionDataset(Dataset):
    def __init__(self, split_set='train', num_points=50000,
                 use_color=False, use_height=False, use_v1=False,
                 augment=False, scan_idx_list=None, ROOT_DIR = None,
                 latent_code_dim = 0, layout_dataset_flag = True):

        self.depth_path = os.path.join(ROOT_DIR,'igibson_depth')
        # self.depth_path = os.path.join(ROOT_DIR,'igibson_depth_panoformer')
        self.bbox_path = os.path.join(ROOT_DIR,'igbson_3dbbox')

        self.cor_dir = os.path.join(ROOT_DIR, 'label_cor')
        self.cor_occ_dir = os.path.join(ROOT_DIR, 'label_cor_occ')
        self.outside_label_map = os.path.join(ROOT_DIR,'outside_label_map')

        if(latent_code_dim==128 or latent_code_dim==0):
            self.obj_emb_path = os.path.join(ROOT_DIR,'igbson_obj_emb')# pc-AE
        elif(latent_code_dim==512):
            if (torch.cuda.is_available()):
                self.obj_emb_path = os.path.join(ROOT_DIR, 'igibson_occ_emb_20221028')  # occ-AE
            else:
                self.obj_emb_path = os.path.join(ROOT_DIR,'igibson_occ_emb') # occ-AE
        else:
            print("latent_code_dim error!")
            exit()
            
        self.image_path = os.path.join(ROOT_DIR,"igbison_image")
        if (torch.cuda.is_available()):
            self.point_label_path = os.path.join(ROOT_DIR, "igbson_3d_labelmap_seg")
        else:
            self.point_label_path = os.path.join(ROOT_DIR,"igbson_3d_labelmap")

        if split_set == "train":
            self.scan_names = self.read_pkl_list(os.path.join(ROOT_DIR,'train.json'))
        elif split_set == "test":
            self.scan_names = self.read_pkl_list(os.path.join(ROOT_DIR, 'test.json'))
        elif split_set == "val":
            self.scan_names = self.read_pkl_list(os.path.join(ROOT_DIR, 'test.json'))
        else:
            exit()

        if scan_idx_list is not None:
            self.scan_names = [self.scan_names[i] for i in scan_idx_list]
        self.num_points = num_points
        self.augment = augment
        self.use_color = use_color
        self.use_height = use_height
        self.latent_code_dim = latent_code_dim
        self.split_set = split_set

        self.fibonacci_mask = self.get_fibonacci_mask(nb_samples=60000)
        self.unit_map = self.get_unit_map()
        self.sample_pts = self.get_sample_pts()

        initial_mesh_path = ''
        if(os.path.exists(initial_mesh_path)):
            print("Offset Using ",os.path.split(initial_mesh_path)[-1])
            verts, faces, aux = load_obj(initial_mesh_path)
            self.initial_mesh_verts = np.array(verts)
            self.initial_mesh_faces = faces.verts_idx
        else:
            print("Offset Using ICOSPHERE3")
            initial_mesh = ico_sphere(3)
            self.initial_mesh_verts = np.array(initial_mesh.verts_packed())
            self.initial_mesh_faces = initial_mesh.faces_packed()

        self.renderer = getTopViewRenderParms()
        self.renered_mesh_image_flag = False
        self.e2c = Equirec2Cube(512, 1024, 256)

        self.layout_dataset_flag = layout_dataset_flag
        if(self.layout_dataset_flag):
            self.layout_dataset = LayoutDataset()

    def get_sample_pts(self):
        point_cloud = self.unit_map[self.fibonacci_mask]
        pc_uv_normlized = []
        for pc_idx in range(point_cloud.shape[0]):
            current_xyz = point_cloud[pc_idx,:3]
            uv = self.unitxyz2uv(current_xyz, equ_w=1024, equ_h=512, normlized=True)
            pc_uv_normlized.append(np.array([(1-uv[0])-0.5, uv[1]-0.5]).squeeze()*2)
        samp_pts = np.array(pc_uv_normlized).astype(np.float32)
        return samp_pts


    def get_unit_map(self):
        h = 512
        w = 1024
        Theta = np.arange(h).reshape(h, 1) * np.pi / h + np.pi / h / 2
        Theta = np.repeat(Theta, w, axis=1)
        Phi = np.arange(w).reshape(1, w) * 2 * np.pi / w + np.pi / w - np.pi
        Phi = -np.repeat(Phi, h, axis=0)

        X = np.expand_dims(np.sin(Theta) * np.sin(Phi),2)
        Y =  np.expand_dims(np.cos(Theta),2)
        Z = np.expand_dims(np.sin(Theta) * np.cos(Phi),2)
        unit_map = np.concatenate([X,Z,Y],axis=2)

        return unit_map

    def get_fibonacci_mask(self,nb_samples):
        sampled_point = self.fibonacci_spiral_samples_on_unit_sphere(nb_samples=nb_samples)
        mask = np.zeros((512,1024))
        for sampled_point_i in range(sampled_point.shape[0]):
            xyz = sampled_point[sampled_point_i, :]
            uv = self.unitxyz2uv(xyz, equ_w=1024, equ_h=512)
            mask[int(uv[1]), int(uv[0])] = 1
        mask = np.where(mask > 0.5, True, False)
        return mask

    def unitxyz2uv(self, xyz, equ_w, equ_h, normlized = False):
        x, z, y = np.split(xyz, 3, axis=-1)
        lon = np.arctan2(x, z)
        c = np.sqrt(x ** 2 + z ** 2)
        lat = np.arctan2(y, c)

        # longitude and latitude to equirectangular coordinate
        if normlized:
            u = (lon / (2 * np.pi) + 0.5)
            v = (-lat / np.pi + 0.5)
        else:
            u = (lon / (2 * np.pi) + 0.5) * equ_w - 0.5
            v = (-lat / np.pi + 0.5) * equ_h - 0.5
        return [u, v]

    def fibonacci_spiral_samples_on_unit_sphere(self, nb_samples, mode=0):
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

    def read_pkl_list(self, list_file):
        import json
        pkl_list = []
        with open(list_file) as f:
            lines = json.load(f)
            for line in lines:
                pkl_name = line.split("/")
                pkl_list.append(pkl_name[0]+"_"+pkl_name[1])
        return pkl_list

    def vis_save_grid_rgb(self, image, sample_pts, point_cloud):
        image = torch.from_numpy(image.astype(np.float32)/255.).permute(2,0,1).unsqueeze(0)
        rgb_sampled = torch.nn.functional.grid_sample(image, sample_pts.unsqueeze(0)).squeeze().numpy().transpose()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:,:3])
        pcd.colors = o3d.utility.Vector3dVector(rgb_sampled)
        o3d.visualization.draw([pcd])

    def __len__(self):
        return len(self.scan_names)

    def get_gt_mesh_from_cor(self, cor):
        W, H = 1024, 512
        N = len(cor) // 2
        floor_z = -1.6
        floor_xy = np_coor2xy(cor[1::2], floor_z, W, H, floorW=1, floorH=1)
        c = np.sqrt((floor_xy ** 2).sum(1))
        v = np_coory2v(cor[0::2, 1], H)
        ceil_z = (c * np.tan(v)).mean()

        polygon = Polygon(floor_xy.tolist())
        transformation = np.eye(4)
        transformation[0, 0] = -1
        transformation[1, 1] = -1
        transformation[2, 3] = -1.6

        mesh = trimesh.creation.extrude_polygon(polygon, height=ceil_z - floor_z, transform=transformation)

        gt_verts = torch.tensor(mesh.vertices, dtype=torch.float32)
        gt_faces = torch.tensor(mesh.faces)

        return gt_verts, gt_faces, mesh

    def generate_initial_mesh_location(self, gt_trimesh):

        ray_origins = np.zeros((self.initial_mesh_verts.shape[0], 3))
        ray_directions = self.initial_mesh_verts.copy()

        # Get the intersections
        locations, index_ray, index_tri = gt_trimesh.ray.intersects_location(
            ray_origins=ray_origins, ray_directions=ray_directions, multiple_hits=False)

        sorted_locations = np.zeros((self.initial_mesh_verts.shape[0], 3),dtype=np.float32)
        for i in range(locations.shape[0]):
            sorted_locations[index_ray[i], :] = locations[i, :]

        return sorted_locations

    def __getitem__(self, idx):
        """
        Returns a dict with following keys:
            point_clouds: (N,3+C)
            center_label: (MAX_NUM_OBJ,3) for GT box center XYZ
            heading_class_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_HEADING_BIN-1
            heading_residual_label: (MAX_NUM_OBJ,)
            size_classe_label: (MAX_NUM_OBJ,) with int values in 0,...,NUM_SIZE_CLUSTER
            size_residual_label: (MAX_NUM_OBJ,3)
            sem_cls_label: (MAX_NUM_OBJ,) semantic class index
            box_label_mask: (MAX_NUM_OBJ) as 0/1 with 1 indicating a unique box
            vote_label: (N,9) with votes XYZ (3 votes: X1Y1Z1, X2Y2Z2, X3Y3Z3)
                if there is only one vote than X1==X2==X3 etc.
            vote_label_mask: (N,) with 0/1 with 1 indicating the point
                is in one of the object's OBB.
            scan_idx: int scan index in scan_names list
            max_gt_bboxes: unused
        """

        # if(self.split_set == 'train'):
            # current_mask_shift = np.random.randint(low = 0, high = self.fibonacci_mask.shape[0])
            # current_mask = np.roll(self.fibonacci_mask.copy(), current_mask_shift)
        # else:
        current_mask = self.fibonacci_mask.copy()

        scan_name = self.scan_names[idx]
        depth = np.load(os.path.join(self.depth_path, scan_name) + '_depth_pred.npy')  # Nx6
        bboxes = np.load(os.path.join(self.bbox_path, scan_name) + '_bbox.npy')  # K,8
        label_map = np.load(os.path.join(self.point_label_path, scan_name) + '_label_map.npz')['label_map']
        rgb = cv2.imread(os.path.join(self.image_path, scan_name) + '.png')[:, :, (2, 1, 0)]
        outside_label_map = np.expand_dims(np.load(os.path.join(self.outside_label_map,scan_name)+"_outside_map.npy"),2)
        with open(os.path.join(self.cor_dir, scan_name + ".txt")) as f:
            cor_no_occ = np.array([line.strip().split() for line in f if line.strip()], np.float32)
        with open(os.path.join(self.cor_occ_dir, scan_name + ".txt")) as f:
            cor = np.array([line.strip().split() for line in f if line.strip()], np.float32)

        if self.augment:
            # flip
            if np.random.random() > 0.5:
                cor[:, 0] = rgb.shape[1] - 1 - cor[:, 0]
                cor_no_occ[:, 0] = rgb.shape[1] - 1 - cor_no_occ[:, 0]
                rgb = np.flip(rgb, axis=1)
                depth = np.flip(depth, axis=1)
                label_map = np.flip(label_map, axis=1)
                bboxes[:, 0] = -1 * bboxes[:, 0]
                bboxes[:, 6] = - bboxes[:, 6]

            # rotation
            random_choice = np.random.randint(4)
            image_random_rot_list = [0, 0.25, 0.5, 0.75]
            random_rot_list = [0, np.pi/2, np.pi, np.pi/2*3]
            dx_p = image_random_rot_list[random_choice]
            rot_angle = random_rot_list[random_choice]
            dx = int(rgb.shape[1] * dx_p)
            depth = np.roll(depth, dx, axis=1)
            rgb = np.roll(rgb, dx, axis=1)
            label_map = np.roll(label_map, dx, axis=1)
            cor[:, 0] = (cor[:, 0] + dx) % rgb.shape[1]
            cor_no_occ[:, 0] = (cor_no_occ[:, 0] + dx) % rgb.shape[1]
            rot_mat = sunrgbd_utils.rotz(rot_angle)
            bboxes[:, 0:3] = np.dot(bboxes[:, 0:3], np.transpose(rot_mat))
            bboxes[:, 6] -= rot_angle

        # generate colored point cloud
        point_cloud_map = np.repeat(np.expand_dims(depth,axis=2),3,axis=2)*self.unit_map
        point_cloud = point_cloud_map[current_mask]

        if self.use_color:
            point_cloud_rgb = rgb[current_mask]/255.
            point_cloud_rgb = (point_cloud_rgb-MEAN_COLOR_RGB)/MEAN_COLOR_STD
            point_cloud = np.concatenate([point_cloud,point_cloud_rgb],axis=1)

        # generate votes
        point_labels = label_map[current_mask]
        point_obj_mask = point_labels[:, 0]
        point_instance_label = point_labels[:, 1]
        outside_labels = outside_label_map[current_mask][:,0]

        # ------------------------------- LABELS ------------------------------
        box3d_centers = np.zeros((MAX_NUM_OBJ, 3))
        box3d_sizes = np.zeros((MAX_NUM_OBJ, 3))
        angle_classes = np.zeros((MAX_NUM_OBJ,))
        angle_residuals = np.zeros((MAX_NUM_OBJ,))
        size_classes = np.zeros((MAX_NUM_OBJ,))
        size_residuals = np.zeros((MAX_NUM_OBJ, 3))
        label_mask = np.zeros((MAX_NUM_OBJ))
        label_mask[0:bboxes.shape[0]] = 1
        max_bboxes = np.zeros((MAX_NUM_OBJ, 8))
        max_bboxes[0:bboxes.shape[0], :] = bboxes[:,:8]

        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            semantic_class = bbox[7]
            box3d_center = bbox[0:3]
            angle_class, angle_residual = DC.angle2class(bbox[6])
            # NOTE: The mean size stored in size2class is of full length of box edges,
            # while in sunrgbd_data.py data dumping we dumped *half* length l,w,h.. so have to time it by 2 here
            box3d_size = (bbox[3:6]) * 2
            size_class, size_residual = DC.size2class(box3d_size, DC.class2type[semantic_class])
            box3d_centers[i, :] = box3d_center
            angle_classes[i] = angle_class
            angle_residuals[i] = angle_residual
            size_classes[i] = size_class
            size_residuals[i] = size_residual
            box3d_sizes[i, :] = box3d_size

        target_bboxes_mask = label_mask
        target_bboxes = np.zeros((MAX_NUM_OBJ, 6))
        target_bboxes[:, 0:3] += 1000.0
        size_gts = np.zeros((MAX_NUM_OBJ, 3))
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            corners_3d = sunrgbd_utils.my_compute_box_3d(bbox[0:3], bbox[3:6], bbox[6])
            # compute axis aligned box
            xmin = np.min(corners_3d[:, 0])
            ymin = np.min(corners_3d[:, 1])
            zmin = np.min(corners_3d[:, 2])
            xmax = np.max(corners_3d[:, 0])
            ymax = np.max(corners_3d[:, 1])
            zmax = np.max(corners_3d[:, 2])
            target_bbox = np.array(
                [(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2, xmax - xmin, ymax - ymin, zmax - zmin])
            target_bboxes[i, :] = target_bbox
            size_gts[i, :] = target_bbox[3:6]

        point_cloud, choices = pc_util.random_sampling(point_cloud, self.num_points, return_choices=True)
        sample_pts = torch.from_numpy(self.sample_pts[choices]).contiguous().unsqueeze(0)
        # self.vis_save_grid_rgb(rgb, sample_pts, point_cloud)
        point_obj_mask = point_obj_mask[choices]
        point_instance_label = point_instance_label[choices]
        outside_labels = outside_labels[choices]

        ret_dict = {}
        ret_dict['point_clouds'] = point_cloud.astype(np.float32)
        ret_dict['center_label'] = target_bboxes.astype(np.float32)[:, 0:3]
        ret_dict['heading_class_label'] = angle_classes.astype(np.int64)
        ret_dict['heading_residual_label'] = angle_residuals.astype(np.float32)
        ret_dict['size_class_label'] = size_classes.astype(np.int64)
        ret_dict['size_residual_label'] = size_residuals.astype(np.float32)
        
        # 多的
        # ret_dict['size_gts'] = size_gts.astype(np.float32)
        ret_dict['size_gts'] = box3d_sizes.astype(np.float32)

        target_bboxes_semcls = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls[0:bboxes.shape[0]] = bboxes[:, 7]  # from 0 to 9

        sem_weight = np.ones(bboxes[:, 7].shape, dtype = np.float32)
        sem_weight[np.where(bboxes[:, 7]==18)] = 10
        target_bboxes_semcls_weight = np.zeros((MAX_NUM_OBJ))
        target_bboxes_semcls_weight[0:bboxes.shape[0]] = sem_weight
        ret_dict['sem_cls_weight'] = target_bboxes_semcls_weight

        ret_dict['sem_cls_label'] = target_bboxes_semcls.astype(np.int64)
        ret_dict['box_label_mask'] = target_bboxes_mask.astype(np.float32)

        # 不同的
        ret_dict['point_obj_mask'] = point_obj_mask.astype(np.int64)
        ret_dict['outside_label_mask'] = outside_labels.astype(np.int64)
        ret_dict['point_instance_label'] = point_instance_label.astype(np.int64)

        ret_dict['scan_idx'] = np.array(idx).astype(np.int64)
        ret_dict['max_gt_bboxes'] = max_bboxes

        latent_codes = np.zeros((MAX_NUM_OBJ,self.latent_code_dim))
        emb = np.load(os.path.join(self.obj_emb_path, scan_name) + '_embs.npy')
        latent_codes[:bboxes.shape[0], :] = emb[:,0:(self.latent_code_dim)]
        ret_dict['max_gt_latent_codes'] = latent_codes.astype(np.float32)

        # for image_feature fusion
        image = torch.from_numpy(((rgb[:,:, :3]/ 255. - MEAN_COLOR_RGB) / MEAN_COLOR_STD).astype(np.float32)).permute((2, 0, 1))
        ret_dict['image'] = image

        cube_rgb = self.e2c.run(rgb[:,:, :3])
        cube_image = torch.from_numpy(((cube_rgb[:,:, :3]/ 255. - MEAN_COLOR_RGB) / MEAN_COLOR_STD).astype(np.float32)).permute((2, 0, 1))
        ret_dict['cube_image'] = cube_image

        ret_dict['sample_pts'] = sample_pts
        ret_dict['scan_name'] = scan_name

        # load layout-dataset
        if self.layout_dataset_flag:
            if True:
                gt_verts, gt_faces, gt_mesh = self.get_gt_mesh_from_cor(cor)
                # save_obj('/Users/yuandong/Documents/Git_project_DAMO/gp3d_private_HorizonNet/igibson/vis_output_dir/'+scan_name+'_gtmesh.obj', gt_verts, gt_faces)
                meshes_gt = Meshes(verts=[gt_verts], faces=[gt_faces])
                if self.renered_mesh_image_flag:
                    gt_rendered_image = generateMeshRenderImage(meshes_gt,renderer=self.renderer)
                    ret_dict['gt_mesh_render_image'] = gt_rendered_image

                initial_mesh_loc = self.generate_initial_mesh_location(gt_trimesh = gt_mesh)
                ret_dict['initial_mesh_loc'] = torch.from_numpy(initial_mesh_loc)

                points_gt, normals_gt = sample_points_from_meshes(meshes_gt, num_samples=5000, return_normals=True)
                edge_points_gt = sample_points_from_edges(meshes_gt, no_sampling=False)
                ret_dict['points_gt'] = points_gt.squeeze(0)
                ret_dict['normals_gt'] = normals_gt.squeeze(0)
                ret_dict['edge_points_gt'] = edge_points_gt.squeeze(0)

            layout_dict = self.layout_dataset.get_layout_data(cor_no_occ)
            for key in layout_dict:
                ret_dict[key] = layout_dict[key]

        # obj-token、layout-token、point-token、image-token
        num_tokens = [256, 642, 1024]
        mjm_masks = []
        mvm_percent = 0.1
        for num_token in num_tokens:
            pb = np.random.random_sample()
            mjm_mask = np.ones((num_token, 1))
            masked_num = int(pb * mvm_percent * num_token)  # at most x% of the joints could be masked
            indices = np.random.choice(np.arange(num_token), replace=False, size=masked_num)
            mjm_mask[indices, :] = 0.0
            mjm_mask = torch.from_numpy(mjm_mask).float()
            mjm_masks.append(mjm_mask)
        mjm_masks = torch.cat(mjm_masks,0)
        ret_dict['meta_masks'] = mjm_masks
        return ret_dict


def viz_votes(pc, point_votes, point_votes_mask):
    """ Visualize point votes and point votes mask labels
    pc: (N,3 or 6), point_votes: (N,9), point_votes_mask: (N,)
    """
    inds = (point_votes_mask == 1)
    pc_obj = pc[inds, 0:3]
    pc_obj_voted1 = pc_obj + point_votes[inds, 0:3]
    pc_obj_voted2 = pc_obj + point_votes[inds, 3:6]
    pc_obj_voted3 = pc_obj + point_votes[inds, 6:9]
    pc_util.write_ply(pc_obj, 'pc_obj.ply')
    pc_util.write_ply(pc_obj_voted1, 'pc_obj_voted1.ply')
    pc_util.write_ply(pc_obj_voted2, 'pc_obj_voted2.ply')
    pc_util.write_ply(pc_obj_voted3, 'pc_obj_voted3.ply')


def viz_obb(pc, label, mask, angle_classes, angle_residuals,
            size_classes, size_residuals, ouput_dir):
    """ Visualize oriented bounding box ground truth
    pc: (N,3)
    label: (K,3)  K == MAX_NUM_OBJ
    mask: (K,)
    angle_classes: (K,)
    angle_residuals: (K,)
    size_classes: (K,)
    size_residuals: (K,3)
    """
    oriented_boxes = []
    K = label.shape[0]
    for i in range(K):
        if mask[i] == 0: continue
        obb = np.zeros(7)
        obb[0:3] = label[i, 0:3]
        heading_angle = DC.class2angle(angle_classes[i], angle_residuals[i])
        box_size = DC.class2size(size_classes[i], size_residuals[i])
        obb[3:6] = box_size
        obb[6] = -1 * heading_angle
        print(obb)
        oriented_boxes.append(obb)
    pc_util.write_oriented_bbox(oriented_boxes, os.path.join(ouput_dir,'gt_obbs.ply'))
    pc_util.write_ply(label[mask == 1, :], os.path.join(ouput_dir,'gt_centroids.ply'))


def get_sem_cls_statistics():
    """ Compute number of objects for each semantic class """
    d = IGbsonDetectionDataset(use_height=True, use_color=True, use_v1=True, augment=True)
    sem_cls_cnt = {}
    for i in range(len(d)):
        if i % 10 == 0: print(i)
        sample = d[i]
        pc = sample['point_clouds']
        sem_cls = sample['sem_cls_label']
        mask = sample['box_label_mask']
        for j in sem_cls:
            if mask[j] == 0: continue
            if sem_cls[j] not in sem_cls_cnt:
                sem_cls_cnt[sem_cls[j]] = 0
            sem_cls_cnt[sem_cls[j]] += 1
    print(sem_cls_cnt)


def loadOCCNetGenerator(model_filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder_type = 'pointnet_resnet'
    decoder_type = 'cbatchnorm'
    input_dim = 3
    z_dim = 0
    latentcode_dim = 512

    encoder_kwargs = {'hidden_dim': 512}

    encoder_latent = None

    if encoder_type is not None:
        encoder = encoder_dict[encoder_type](
            c_dim=latentcode_dim,
            **encoder_kwargs)
    else:
        encoder = None

    def get_prior_z(z_dim, device, **kwargs):
        ''' Returns prior distribution for latent code z.

        Args:
            z_dim : conditioned laten code dimension
            device (device): pytorch device
        '''
        p0_z = dist.Normal(
            torch.zeros(z_dim, device=device),
            torch.ones(z_dim, device=device)
        )

        return p0_z
    
    p0_z = get_prior_z(z_dim, device)
    decoder = decoder_dict[decoder_type](dim=input_dim, z_dim=z_dim, c_dim=latentcode_dim)
    model = OccupancyNetwork(decoder, encoder, p0_z=p0_z, device=device)
    checkpoint_io = CheckpointIO(checkpoint_dir=BASE_DIR, model=model)
    checkpoint_io.load(model_filepath)

    generator = Generator3D(model, device=device)
    return generator

def write_wireframe_bboxes(heading_class_label,heading_residual_label,center_label,
                           size_class_label,size_residual_label,box_label_mask,
                           sem_cls_label, output_dir):
    import trimesh

    x_corners = [-1 / 2, 1 / 2, 1 / 2, -1 / 2, -1 / 2, 1 / 2, 1 / 2, -1 / 2]
    y_corners = [1 / 2, 1 / 2, -1 / 2, -1 / 2, 1 / 2, 1 / 2, -1 / 2, -1 / 2]
    z_corners = [1 / 2, 1 / 2, 1 / 2, 1 / 2, -1 / 2, -1 / 2, -1 / 2, -1 / 2]
    shape_out = np.vstack([x_corners, y_corners, z_corners]).transpose()

    all_obj_vertices = []
    for i in range(heading_class_label.shape[0]):
        if box_label_mask[i] == 0: continue
        rotation_matrix = heading2rotmat(
            -DC.class2angle(heading_class_label[i], heading_residual_label[i]))
        translation = center_label[i, 0:3]
        size = DC.class2size(size_class_label[i], size_residual_label[i])
        obj_vertices = (rotation_matrix @ (shape_out * size).T).T + translation
        all_obj_vertices.append(obj_vertices)

    radius = 0.025
    mesh = []
    colorbox = DC.get_colorbox()
    layout_lines = [[0,1],[0,3],[0,4],[1,2],[1,5],[2,3],[2,6],[3,7],[4,5],[4,7],[5,6],[6,7]]
    for idx in range(len(all_obj_vertices)):
        corners_3d = all_obj_vertices[idx]
        for indexes in layout_lines:
            line = corners_3d[indexes]
            line_mesh = trimesh.creation.cylinder(radius, sections=8, segment=line)
            line_mesh.visual.vertex_colors[:] = np.append(colorbox[sem_cls_label[idx]], 255).astype(np.uint8)
            mesh.append(line_mesh)
    mesh = sum(mesh)
    trimesh.exchange.export.export_mesh(mesh, os.path.join(output_dir,'merged_wireframe_bboxes.ply'))

def heading2rotmat(heading_angle):
    pass
    rotmat = np.zeros((3, 3))
    rotmat[2, 2] = 1
    cosval = np.cos(heading_angle)
    sinval = np.sin(heading_angle)
    rotmat[0:2, 0:2] = np.array([[cosval, -sinval], [sinval, cosval]])
    return rotmat

if __name__ == '__main__':
    vis_pc_autoencoder_flag = False
    merge_pc = True

    if(vis_pc_autoencoder_flag):
        latent_code_dim = 128
    else:
        latent_code_dim = 0

    d = IGbsonDetectionDataset(split_set='train',
                               num_points=50000,
                               use_height=False,
                               use_color=True,
                               use_v1=True,
                               augment=True,
                               ROOT_DIR= "/Users/yuandong/Documents/Git_project_DAMO/gp3d_private/igibson/example_data",
                               latent_code_dim = latent_code_dim,
                               layout_dataset_flag=True)

    sample = d[0]

    vis_output_dir = "./vis_output_dir"
    os.makedirs(vis_output_dir,exist_ok=True)

    # write_ply
    pc_util.write_ply(sample['point_clouds'], os.path.join(vis_output_dir,'pc.ply'))
    pc_util.write_ply(sample['points_gt'], os.path.join(vis_output_dir,'points_gt.ply'))
    pc_util.write_ply(sample['edge_points_gt'], os.path.join(vis_output_dir,'edge_points_gt.ply'))

    # write_bbox
    viz_obb(sample['point_clouds'], sample['center_label'], sample['box_label_mask'],
            sample['heading_class_label'], sample['heading_residual_label'],
            sample['size_class_label'], sample['size_residual_label'],vis_output_dir)

    write_wireframe_bboxes(sample['heading_class_label'],sample['heading_residual_label'],sample['center_label'],
                           sample['size_class_label'],sample['size_residual_label'],sample['box_label_mask'],
                           sample['sem_cls_label'],vis_output_dir)


    exit()


    test_loader = torch.utils.data.DataLoader(d,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True,
                                              drop_last=False)
    for batch_idx, batch_data_label in enumerate(test_loader):
        from module.layout.private_loss import layout_3d_loss

        loss_3d = layout_3d_loss.layout_3dloss()

        gt_lonlat_up = torch.cat([batch_data_label['unit_lonlat'][:, :, 0:1], batch_data_label['bon'][:, 0, :, None]], dim=-1)
        gt_lonlat_down = torch.cat([batch_data_label['unit_lonlat'][:, :, 0:1], batch_data_label['bon'][:, 1, :, None]], dim=-1)

        pred_up_xyz, GT_up_xyz = loss_3d.lonlat2xyz_up(gt_lonlat_up, gt_lonlat_up, batch_data_label['ratio'])
        GT_down_xyz, _ = loss_3d.lonlat2xyz_down(gt_lonlat_down)

        pcd = o3d.geometry.PointCloud()
        layout_pc = np.array(pred_up_xyz.squeeze())
        pcd.points = o3d.utility.Vector3dVector(np.array([-layout_pc[:,0],layout_pc[:,2],-layout_pc[:,1]]).transpose())
        pcd.colors = o3d.utility.Vector3dVector(np.ones((pred_up_xyz.squeeze()).shape)*125)

        pcd3 = o3d.geometry.PointCloud()
        layout_pc2 = np.array(GT_down_xyz.squeeze())
        pcd3.points = o3d.utility.Vector3dVector(np.array([-layout_pc2[:,0],layout_pc2[:,2],-layout_pc2[:,1]]).transpose())
        pcd3.colors = o3d.utility.Vector3dVector(np.ones((GT_down_xyz.squeeze()).shape)*0.5)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(np.array(sample['point_clouds'][:,:3].squeeze()))

        o3d.io.write_point_cloud("/Users/yuandong/Documents/Git_project_DAMO/gp3d_private_HorizonNet/igibson/vis_output_dir/up.ply",pcd)
        o3d.io.write_point_cloud("/Users/yuandong/Documents/Git_project_DAMO/gp3d_private_HorizonNet/igibson/vis_output_dir/down.ply", pcd3)

        o3d.visualization.draw([pcd,pcd3])

        break


    # write decoded-pc / decoded-mesh
    if (vis_pc_autoencoder_flag):
        import torch
        from igibson.example_data.auto_encoder import PointCloudAE

        if latent_code_dim == 128:
            model_path = "example_data/ig_autoencoder.pth"
            estimator = PointCloudAE(128, 2048)
            estimator.load_state_dict(torch.load(model_path, map_location='cpu'))
            estimator.eval()
        elif latent_code_dim == 512:
            from models.OccNet.model import OccupancyNetwork, decoder_dict
            from models.OccNet.generation import Generator3D
            from models.OccNet.im2mesh.encoder import encoder_dict
            from models.OccNet.checkpoints import CheckpointIO
            model_path = "example_data/ig_occnet.pt"
            estimator = loadOCCNetGenerator(model_path)

        merged_pc = []
        for i in range(sample['box_label_mask'].shape[0]):
            if sample['box_label_mask'][i] == 0: continue
            latent_code = sample['max_gt_latent_codes'][i,:]
            with torch.no_grad():
                emb = torch.FloatTensor(latent_code).unsqueeze(0)
                if latent_code_dim == 128:
                    with torch.no_grad():
                        _, shape_out = estimator(None, emb)
                    shape_out = shape_out.squeeze().detach().numpy()
                elif latent_code_dim == 512:
                    shape_out = estimator.generate_mesh_from_latent(emb)

            rotation_matrix = heading2rotmat(-DC.class2angle(sample['heading_class_label'][i], sample['heading_residual_label'][i]))
            translation = sample['center_label'][i,0:3]
            size = DC.class2size(sample['size_class_label'][i], sample['size_residual_label'][i])
            if latent_code_dim == 128:
                obj_vertices = (rotation_matrix @ (shape_out * size).T).T + translation
                if merge_pc:
                    merged_pc.append(obj_vertices)
                else:
                    pc_util.write_ply(obj_vertices, os.path.join(vis_output_dir,'obj'+str(i)+'.ply'))
            elif latent_code_dim == 512:
                shape_out.apply_scale(size)
                T = np.identity(4)
                T[:3,:3] = rotation_matrix
                T[:3, 3] = translation
                shape_out.apply_transform(T)
                shape_out.export(os.path.join(vis_output_dir,'obj'+str(i)+'.ply'))

        if latent_code_dim==128 and merge_pc:
            merged_pc = np.concatenate(merged_pc,0)
            pc_util.write_ply(merged_pc, os.path.join(vis_output_dir, 'merged_pc.ply'))
            # merged_pc = np.array([-merged_pc[:,0], -merged_pc[:,2], merged_pc[:,1]]).transpose()
            # pc_util.write_ply(merged_pc, os.path.join(vis_output_dir, sample['scan_name']+'_pc.ply'))